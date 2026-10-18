import tilelang
import tilelang.language as T

# 朴素 GEMM：不做任何性能优化，作为 profiling 的基线。
# 沿 K 维的循环直接从 global memory 读写，没有 shared memory、没有流水、没有 Tensor Core。
# 累加器也不用寄存器：就是 global memory 里的 C 元素本身。
#
# M、N、K、block_M、block_N 是编译期常量，调用时确定。
# 2048³、block 128×128 是一组配置，换配置就重新编译。
def gemm_naive(M, N, K, block_M, block_N):
    # @T.prim_func：main 不被 Python 执行，被解析成编译器的 IR。
    # 函数体是给编译器读的数据流描述，不是给 GPU 跑的指令流。
    @T.prim_func
    def main(
        # 签名：三个二维 tensor，形状 (M, K)、(K, N)、(M, N)，dtype 都是 float32。
        # 形状和 dtype 是全部类型信息，编译器据此推导访存的类型、宽度、对齐。
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        # T.Kernel 声明归属，两个参数各一层：
        # 第一个参数（二维 grid）：C 切成 (M/block_M) × (N/block_N) 个子块，
        #   切出多少子块就声明多少 block，第 (by, bx) 个 block 负责 C 的第 (by, bx) 个子块；
        # threads=128：每个 block 128 个线程。
        # 本配置下 grid 是 16×16，每个 block 负责 C 的一个 128×128 子块。
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # T.Parallel 声明可并行的循环。这里是两层循环 block_M × block_N，
            # 共 128×128 次迭代，摊到 128 个线程上执行——哪个线程跑哪次迭代由编译器推导。
            # 循环体里没有线程概念：i、j 是迭代编号，不是线程编号。
            for i, j in T.Parallel(block_M, block_N):
                # 每个线程负责 C 的一个元素，沿 K 维串行乘加 K 次。
                # T.serial 是普通串行循环：K 次迭代按顺序执行，每次依赖前一次的结果。
                # 累加器就是 global memory 里的 C 元素本身：
                # 每次迭代读 A、B 各一个元素，再对 C 做一次读-改-写，全部访问 global memory。
                for k in T.serial(K):
                    # 索引拆开看：
                    #   by * block_M + i —— block 负责的 C 子块的起始行 + 子块内第 i 行
                    #   bx * block_N + j —— 起始列 + 子块内第 j 列
                    #   k —— 沿 K 维的第 k 次乘加
                    # 整行读作：线程 (i, j) 的第 k 次迭代，
                    # C[by*block_M+i, bx*block_N+j] += A 的第 by*block_M+i 行第 k 列 × B 的第 k 行第 bx*block_N+j 列。
                    # C 元素乘加 K 次后就是点积 A 的第 (by*block_M+i) 行 · B 的第 (bx*block_N+j) 列。
                    C[by * block_M + i, bx * block_N + j] += A[by * block_M + i, k] * B[k, bx * block_N + j]

    return main

if __name__ == "__main__":
    M = N = K = 2048
    # 编译：Parser → IR Builder → Optimization → Codegen → 加载。
    # out_idx=[2] 声明 C 是输出，wrapper 会为 C 分配显存并返回。
    kernel = tilelang.compile(
        gemm_naive(M, N, K, block_M=128, block_N=128),
        out_idx=[2], target="cuda")

    import torch
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = kernel(a, b)
    # 正确性校验：L2 范数比。
    # naive 的乘加由 CUDA core（GPU 的通用浮点单元）执行，是完整的 fp32 精度，误差应为 0。
    ref = a @ b
    rel_err = ((c - ref).norm() / ref.norm()).item()
    assert rel_err < 1e-3, rel_err
    print("gemm naive ok: rel_err = %.2e" % rel_err)

    # 计时：预热 3 次去掉一次性初始化开销；
    # CUDA event 在 GPU stream 上打时间戳，只对齐计算时间（launch 是异步的）；
    # 测 10 次取均值，去掉时钟噪声。
    for _ in range(3):
        kernel(a, b)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(10):
        kernel(a, b)
    t1.record(); torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / 10
    # TFLOPS = 运算量 / 时间。GEMM 运算量是 2×M×N×K。
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print("time: %.3f ms, %.2f TFLOPS" % (ms, tflops))
