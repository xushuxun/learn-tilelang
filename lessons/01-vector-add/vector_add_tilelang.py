import tilelang
import tilelang.language as T

# prim_func：kernel 描述的容器。
# @T.prim_func 装饰的函数不会被 Python 执行，而是被解析成 IR——
# 这是 TileLang 心智模型的第一条：你写的代码是给编译器读的数据流描述，
# 不是给 GPU 跑的指令流。
#
# n 和 block_size 是编译期常量，来自 host 侧的配置。
# 它们决定数据怎么切分：n 个元素切成多少个 tile、每个 tile 多大。
# 切分方式属于数据流描述，性能直接相关，所以留给程序员决定。
def vector_add(n, block_size):
    @T.prim_func
    def main(
        # 签名声明三个 buffer：形状 (n,) + dtype "float32"。
        # 形状和 dtype 就是全部类型信息——TileLang 据此推导访存的类型、宽度、对齐，
        # 不需要指针，也不需要手动指定元素大小。
        # 这三个 buffer 是"数据流"的输入输出：读 A、B，写 C。
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        # T.Kernel 声明 tile 的归属，两个参数都是显式给出的：
        #   第一个参数：grid 大小 = block 数。T.ceildiv(n, block_size) 向上取整，
        #     保证 n 个元素被全覆盖——这就是"先划数据块"：n 个元素切成
        #     ceildiv(n, block_size) 个 tile，每个 tile block_size 个元素。
        #   threads=block_size：每个 block 的 thread 数。
        # 这两个数字决定硬件行为（grid 维数、每 block 线程数），
        # 是程序员保留的性能决策权。
        # as bx：bx 是当前 block 的坐标（对应 CUDA 的 blockIdx.x），
        #   with 块内的代码是"一个 block 对它名下那个 tile"的数据流描述。
        with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as bx:
            # T.Parallel 是调度原语，不是数据流。
            # 它声明：这个循环有 block_size 次迭代，这些迭代映射到 block 内的
            # 线程上执行——映射关系、谁执行哪次、访存怎么向量化，全由编译器的
            # Layout Inference 推导，代码里不指定。
            # 这里迭代数等于线程数，推断结果是一一对应；
            # 改成 T.Parallel(block_size * 4) 会得到每线程 4 次迭代 + float4 向量化。
            for i in T.Parallel(block_size):
                # 循环体是数据流的定义：这次迭代读写哪里、算什么。
                # bx * block_size 是这个 block 名下 tile 的起始下标，
                # + i 是 tile 内的第 i 个元素。
                # 整行读作：block bx 的第 i 次迭代，把 A、B 对应元素相加写进 C。
                # 没有 threadIdx，没有地址展开——那是 Lower 阶段生成物的事。
                C[bx * block_size + i] = A[bx * block_size + i] + B[bx * block_size + i]

    return main

# 注释：n 必须能被 block_size 整除，因为上面没有写边界保护（if i < n）。
# 非整除时最后一个 tile 会越界读——边界保护属于数据流的一部分，
# 编译器不会替你猜，要么写 if，要么把 n 取整（课程里另一个版本演示过）。
if __name__ == "__main__":
    n, block_size = 1 << 20, 256
    kernel = tilelang.compile(vector_add(n, block_size), out_idx=[2], target="cuda")

    import torch
    a = torch.ones(n, device="cuda", dtype=torch.float32)
    b = torch.ones(n, device="cuda", dtype=torch.float32)
    c = kernel(a, b)
    assert torch.equal(c, 2 * torch.ones(n, device="cuda"))
    print("vector add ok, n =", n)

    print(kernel.get_kernel_source())  # 打印编译生成的 CUDA 源码
