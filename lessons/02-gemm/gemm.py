import tilelang
import tilelang.language as T

# GEMM：C = A @ B，全部用 TileLang 的 tile 算子描述
#
# 切分决策（性能来源，程序员给出）：
#   C 切成 (M/block_M) x (N/block_N) 个 tile，每 tile block_M x block_N
#   每个 tile 内部沿 K 维再切 block_K，一次只搬一个 K 片
def gemm(M, N, K, block_M, block_N, block_K, num_stages):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        # 归属声明：一个 block 负责 C 的一个 block_M x block_N 子块
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 显式内存放置：buffer 放在哪一层，这个决策也是程序员的
            # A/B 的当前 K 片放在 shared memory（快，block 内共享）
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            # 累加器放在寄存器 fragment。
            # fragment：分布在寄存器上的 buffer。C_local 声明为 128×128，但不占一块连续的内存：
            # 它的 16384 个元素按编译器推导的线程映射摊到各线程的寄存器，
            # 128 个线程，每个线程持有 128 个元素，作为该线程的私有累加器。
            # 与 shared memory 的区别：shared 是一块内存，按地址索引、block 内所有线程访问；
            # fragment 没有地址，每个线程只读写分到自己寄存器的那 128 个元素。
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            # 数据流：沿 K 维逐片搬入 -> 片上 GEMM -> 累加
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

if __name__ == "__main__":
    M = N = K = 2048
    # block_M/N/K 和 num_stages 决定生成物的形态
    kernel = tilelang.compile(
        gemm(M, N, K, block_M=128, block_N=128, block_K=32, num_stages=3),
        out_idx=[2], target="cuda")

    import torch
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = kernel(a, b)
    # T.gemm 的乘加由 Tensor Core 执行，指令把 fp32 输入截断成 TF32（尾数 10 位），相对误差约 5e-4。
    # 按相对误差（L2 范数比）校验，比逐元素 allclose 适合
    ref = a @ b
    rel_err = ((c - ref).norm() / ref.norm()).item()
    assert rel_err < 1e-3, rel_err
    print("gemm ok: rel_err = %.2e" % rel_err)

    for _ in range(3):  # 预热
        kernel(a, b)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(10):
        kernel(a, b)
    t1.record(); torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / 10
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print("time: %.3f ms, %.2f TFLOPS" % (ms, tflops))
