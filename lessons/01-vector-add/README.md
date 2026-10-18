# Vector Add ——  TileLang 对 GPU 编程的抽象

目标：读完这一课，能直接读原论文 [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/abs/2504.17577)（北大、微软研究院）。结构：先总结论文，再看手写 CUDA 的问题，然后讲编译器原理，最后用向量加法把论文的概念逐个落地。

## 论文总结

问题。AI kernel 的性能高度依赖手写优化，但手写难。难点不在于数据流——AI kernel 的数据流模式高度雷同，基本都是"在 DRAM 和 SRAM 之间搬 tile，对 tile 做计算"——而在于数据流之外的四项调度工作：线程绑定（tile 操作和数据映射到哪些线程）、内存布局（数据在物理内存里怎么排布以避开 bank conflict、保证合并访存）、张量化（怎么用 Tensor Core 这类硬件指令）、流水线（搬运和计算怎么重叠以藏住访存延迟）。

对现有工具的不满。Triton 这类 DSL 把调度藏在自动策略后面，写简单，但专家想干预时拿不到入口——量化 GEMM 需要内联汇编做类型转换、自定义布局对齐硬件 buffer，在 Triton 里都很别扭。TVM 把计算和调度分开，但调度程序本身难写难懂。

方案。TileLang：数据流和调度解耦。用户用 tile 算子描述数据流；调度由编译器自动推导，推导不够时用户可以拿调度原语覆盖。三条设计原则：Pythonic（嵌在 Python 里）、dataflow-centric、composable（kernel、算子、调度策略可组合）。

效果。FlashAttention 上比 FlashAttention-3、Triton、PyTorch 分别快 1.36×、1.41×、1.70×；Mamba-2 的 chunk 算子比 Triton 平均快 1.77× 到 2.10×；GEMM 达到 cuBLAS 水平。论文的主张：默认调度就能到 SOTA，专家入口又开着。

## 手写 CUDA

C[i] = A[i] + B[i]，N 个元素互不依赖。手写版 `vector_add.cu` 完整代码：

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)  // 边界保护：n 不一定是 block 大小的整数倍
        c[i] = a[i] + b[i];
}

int main() {
    const int n = 1000003;  // 故意用非 256 整数倍
    const int bytes = n * sizeof(float);

    // host 侧分配与初始化
    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { ha[i] = i; hb[i] = 2 * i; }

    // device 侧分配（三次）
    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));

    // host -> device 拷贝（两次）
    CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

    // launch 配置：线程总数 >= n，向上取整
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(da, db, dc, n);
    CHECK(cudaGetLastError());            // launch 是异步的，错误要单独查
    CHECK(cudaDeviceSynchronize());       // 等 kernel 跑完才能拷回结果

    // device -> host 拷贝
    CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));

    // 校验
    for (int i = 0; i < n; ++i)
        if (hc[i] != 3.0f * i) { printf("WRONG at %d: %f\n", i, hc[i]); return 1; }
    printf("vector add ok, n = %d, blocks = %d\n", n, blocks);

    // 三次 device 释放 + 三次 host 释放
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
    free(ha); free(hb); free(hc);
    return 0;
}
```

运行：

```
$ nvcc -O2 -o vector_add vector_add.cu && ./vector_add
vector add ok, n = 1000003, blocks = 3907
```

51 行里，属于数据流的只有 kernel 里的三行（加什么、写到哪里）。其余全是调度：kernel 内的索引公式 `blockIdx.x * blockDim.x + threadIdx.x` 是"哪个 thread 算哪个元素"的绑定；launch 处的 `blocks` 和 `threads` 是 grid 配置；边界保护 `if (i < n)` 是切分不均的善后。host 侧 30 行里大部分是分配、拷贝、同步、释放的样板。

更关键的问题：数据流和调度写在同一行代码区域里。`int i = blockIdx.x * blockDim.x + threadIdx.x` 这一行，前半是调度（线程坐标），后半参与数据流（数组下标），分不开。论文说的"交织"，就是这个。交织的直接后果：换一种切分方式、换一种线程映射，数据流代码要跟着改；编译器也无法介入调度决策，因为决策已经以指令的形式固化了。

TileLang 的解法是把这两类内容拆开，让编译器接管调度。先讲编译器靠什么接管，再讲代码怎么写。

## TileLang 编译器原理

TileLang 编译一个 kernel 走五阶段：Parser（Python AST）→ IR Builder（TVM IR）→ Optimization（推导 buffer 布局和线程绑定）→ Codegen（CUDA/HIP 源码）→ 加载执行。DSL 的每一项设计都能在 Optimization 阶段找到对应的需求。

### tile 是一等对象

论文对 tile 的定义：一块有形状的数据，加上一个指派：这块数据的全部读写由某个并行执行单元（warp、thread block）完成，其他单元不碰它。定义有两部分：

- 形状：哪一块数据。来源是函数签名里的 `T.Tensor((n,), "float32")`。
- 归属：上面说的指派。来源是 `T.Kernel` 的声明。

注意 tile 和 block 是两类东西：tile 是数据的切片，block 是硬件的执行单元，本身互不相关。归属是 tile 定义里的一条指派规则：第 bx 个 tile 的全部读写由第 bx 个 block 执行。这条规则怎么建立，见例子中 T.Kernel 一节的拆解。

"归属"的技术含义是读写责任的划分：第 bx 个 block 负责第 bx 个 tile 的所有 load 和 store。这个划分是定义的一部分，不是调度推导出来的细节。它是编译器判断并行合法性的依据——划分成立，不同 block 写的数据没有交集，编译器就不生成跨 block 的同步代码。这也是 TileLang 和手写 CUDA 的第一个分水岭：手写 CUDA 里这个划分是隐式的，藏在 `blockIdx.x * blockDim.x` 这个索引公式里；TileLang 要求显式声明。原因见下一节——编译器需要静态的归属信息来做线程绑定推导。

### 编译器怎么接管调度

Optimization 阶段根据代码里的声明，推导 buffer 布局和线程绑定，然后生成 CUDA 代码。推导能不能做，取决于代码给了编译器什么。三条规定由此而来。

第一，签名必须静态携带形状和 dtype。编译器生成访存指令时，地址表达式必须是定死的式子，比如 `基址 + 4 * (bx * 256 + i)`，其中 256 是 block_size，4 是 float32 的字节数。这两个数是乘进地址的常数。如果它们到运行期才知道，Codegen 阶段只能把乘法和加法原样生成成运行时运算，编译期推导（向量化、对齐）全部无从谈起。所以形状和切分参数必须在编译前固化，代价是每个形状组合单独编译一个 kernel。

第二，必须有 `T.Kernel` 显式声明 grid 和线程数。编译器要把循环迭代分给线程，得先知道两样东西：这段代码在哪个作用域里有效，以及线程最多有几个。`T.Kernel` 的 grid 参数给出 block 编号的取值范围，threads 参数给出迭代分配的上限。没有这两个数，编译器面对一个循环，不知道它该铺到多少线程上。

第三，数据流代码里不许出现线程概念。编译器接管调度的前提是代码里还留着可推导的东西。如果数据流里写了 threadIdx、写了 float4 宽度，布局和映射已经定死，推导无事可做。所以 DSL 规定：这些硬件概念只能出现在 `T.Kernel` 和调度原语里，数据流代码只写数据坐标。迭代分给哪个线程、访存用多宽，留给编译器决定。

## 完整程序

`vector_add_tilelang.py` 全文：

```python
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
        #   with 块内的代码是"一个 block 对它负责的那个 tile"的数据流描述。
        with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as bx:
            # T.Parallel 是调度原语，不是数据流。
            # 它声明：这个循环有 block_size 次迭代，这些迭代映射到 block 内的
            # 线程上执行——映射关系、谁执行哪次、访存怎么向量化，全由编译器
            # 推导，代码里不指定。
            # 这里迭代数等于线程数，推断结果是一一对应；
            # 改成 T.Parallel(block_size * 4) 会得到每线程 4 次迭代 + float4 向量化。
            for i in T.Parallel(block_size):
                # 循环体是数据流的定义：这次迭代读写哪里、算什么。
                # bx * block_size 是这个 block 负责的那个 tile 的起始下标，
                # + i 是 tile 内的第 i 个元素。
                # 整行读作：block bx 的第 i 次迭代，把 A、B 对应元素相加写进 C。
                # 没有 threadIdx，没有地址展开——那是编译器生成物的事。
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
```

运行：

```
$ python vector_add_tilelang.py
vector add ok, n = 1048576
```

## 逐段解释

### 外层函数：编译期配置从哪来

```python
def vector_add(n, block_size):
    ...
    return main
```

`vector_add` 是一个普通的 Python 函数。它不做任何 GPU 相关的事，只做一件事：接收两个整数，返回一个 `@T.prim_func` 装饰过的函数对象 `main`。

关键点：`n` 和 `block_size` 在定义 `main` 时就已经是确定的整数。`main` 的函数签名、循环边界、索引表达式里都直接写着这两个值。编译器看到的是一个没有任何运行期未知数的静态描述。要换一个 n 或换一个 block_size，就再调一次 `vector_add`，得到一个新的 `main`，再编译一次。

站在编译器一侧，这一步是布局推导的前提。布局是索引到地址的代数映射，本例里是 `i ↦ A 基址 + 4 * (bx * 256 + i)`。映射里的 `256` 来自 block_size，`4` 来自 float32，全是乘进地址表达式的编译期常数。如果 block_size 是运行期值，这个映射就不是代数式，而是一堆运行期运算，Codegen 无法静态生成访存指令。所以形状和切分参数必须在进入编译器前固化，代价是每个形状组合编译一个专门的 kernel。

### @T.prim_func：这段代码不执行，被解析

```python
@T.prim_func
def main(...):
    ...
```

`@T.prim_func` 改变了这个函数被 Python 对待的方式。调用 `vector_add(...)` 返回的 `main` 不会被 Python 解释执行。它被 Parser 阶段读取：Python AST → TileLang 的 IR。函数体里的每个语句变成一个 IR 节点。

由此有一条硬界限：with 块内写的是对编译器的声明，不是指令。它不会跑，没有运行期的"当前 block"在 Python 里切换。`bx` 在运行时的真实身份是 CUDA 的 `blockIdx.x`，只有在生成的 kernel 被 GPU 执行时才存在。

编译器在 IR Builder 阶段得到的是一张数据流图：读哪些 buffer、写哪些 buffer、索引表达式是什么。这张图被原样保留到 Optimization 阶段，是布局推导的工作对象。DSL 设计由此有一条约束：图里的代码不许夹带硬件概念，否则推导没有变换空间。下面的段段都能看到这条约束的落实。

### 函数签名：数据长什么样

```python
def main(
    A: T.Tensor((n,), "float32"),
    B: T.Tensor((n,), "float32"),
    C: T.Tensor((n,), "float32"),
):
```

每个参数注解有两部分：

- `(n,)`：形状。一维，长度 n。注意 n 在这里是具体整数，不是符号变量——这个 kernel 只为长度 n 的数据而编译。
- `"float32"`：dtype。四个字节的浮点。

对照手写 CUDA 的 `const float* __restrict__ A`：指针不携带长度，不携带维度数，编译器对 `a[i]` 的合法性一无所知。`T.Tensor` 的签名带着形状，编译器由此得到访存宽度（一次读几个字节）、对齐信息（tile 起始地址和 block_size 的组合决定能否向量化）、以及索引越界的诊断能力。

这一行同时划定了数据流的边界：A、B 是只读输入，C 是写目标。

### T.Kernel：声明每块数据的归属

```python
with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as bx:
```

这一步同时属于两个层面。对程序员，它是"数据切成多少块、谁来算"的决策声明；对编译器，它是布局推导的两个输入。分开看。

先说清两个词。tile 是数据的切片：A 里第 bx * block_size 到第 bx * block_size + 255 个元素，一段地址连续的内存。block 是硬件的执行单元：GPU 调度器分配的一个线程组。tile 没有执行能力，block 不带数据，二者本来没有任何关系。"第 bx 个 block 负责第 bx 个 tile"这个对应关系不是天生的，是这行代码用两个条件建立起来的：

- 数量相等：第一个参数 `ceildiv(n, block_size)` 让 block 数恰好等于 tile 数。切了几块就派几个 block，一个不剩一个不多。
- 同一个坐标：block 的编号 bx 被直接当作 tile 的编号用——循环体里的 `bx * block_size` 就是用 bx 算出第 bx 个 tile 的起始地址。编号即下标。

两个条件缺一不可。数量相等但坐标不接，block 不知道自己做哪块；坐标接了但数量不等，多出来的 tile 没人做。两个条件都在，对应关系才成立。看生成物时这个关系表现为：`blockIdx.x` 和乘数 256 同时出现在索引公式里，前者是 block 编号，后者是 tile 长度。

程序员视角：第一个参数 `T.ceildiv(n, block_size)` 是 grid 里的 block 数，`ceildiv` 向上取整保证任何 n 下 tile 数够覆盖全部元素；`threads=block_size` 是每个 block 的线程数。两个数字直接变成硬件行为（grid 维数、blockDim.x），它们是性能参数：tile 切多大、开多少 block，决定每 block 的工作量和 SM 占用，所以留在语言里由人决定。

编译器视角，两个参数各声明一层归属，各解决推断的一个问题：

- block 级归属：切出多少个 tile 就声明多少个 block，第 bx 个 block 负责第 bx 个 tile 的全部读写。这给线程绑定推导提供了作用域边界——出了这个块 bx 没有定义，进了这个块 bx 是唯一合法的 block 级坐标。没有这个边界，`C[bx * block_size + i]` 里的 bx 是 grid 坐标还是线程坐标，编译器无从判断，每种解释生成完全不同的代码。
- thread 级归属：`threads=block_size` 给了迭代分配的硬件上限。`T.Parallel(block_size * 4)` 能被推导成每线程 4 次迭代，靠的就是编译器知道线程数恰好是 block_size，1024 除以 256 等于 4。
- 归属声明还决定了并行划分的合法性：两个条件成立，第 bx 个 block 只碰第 bx 个 tile，不同 block 的写互不冲突。编译器据此不生成任何跨 block 同步代码。

`as bx` 使"全局第 bx 个执行单元"和"全局第 bx 个 tile"共用同一个编号。`with` 块进入后，块内代码的隐含前提变了：它描述的是单个 block 对自己负责的那个 tile 的操作，不是对整个张量的操作。用具体数字看：n = 1048576，block_size = 256，则 `T.ceildiv(n, block_size)` = 4096。grid 有 4096 个 block，每个 block 256 个线程，总线程数 1048576，恰好等于元素数——后面会看到这意味着每个线程恰好处理一个元素。

对照手写 CUDA：那里的归属是隐式的，藏在同一个索引公式里，分不开。这里归属是 IR 里显式的节点，编译器直接读它。

### T.Parallel：块内并行交给编译器

```python
for i in T.Parallel(block_size):
```

这是一个循环，但 `T.Parallel` 改变了它的语义：这个循环有 block_size 次迭代，这些迭代之间没有依赖，可以并行执行。"可以并行"只声明迭代空间的可并行性，不指定执行方式：

- 哪次迭代由哪个线程做，不指定。
- 访存按什么宽度向量化，不指定。
- 迭代数大于线程数时怎么展开，不指定。

这就是调度原语的样子：它把一类决策显式让渡给编译器，同时留出名字让人在需要时可以覆盖。这些决策由编译器在 Optimization 阶段做出，推导的边界就是上一步声明的归属。

### 循环体：数据流只有一行

```python
C[bx * block_size + i] = A[bx * block_size + i] + B[bx * block_size + i]
```

这行写的是这次迭代读写哪里、算什么。把索引拆开：

- `bx * block_size`：这个 block 负责的那个 tile 的起始下标。第 bx 个 tile 从全局第 bx * block_size 个元素开始。
- `+ i`：tile 内第 i 个元素。i 来自 T.Parallel 的迭代变量。

整行读作：block bx 的第 i 次迭代，读 A、B 各一个元素，相加写进 C。全 kernel 的数据流就这一句。

这行是全篇受约束最严的一行：不许有 threadIdx，不许有地址展开，不许有访存宽度。`i` 是迭代编号，不是线程编号，它和线程的对应关系已让渡给编译器；`bx * block_size + i` 是数据坐标，不是地址运算。约束的存在是刻意的：访存宽度留给编译器推导，`T.Parallel(block_size * 4)` 能推出 float4 向量化，靠的就是循环体里没有任何东西把宽度固定下来。如果这里写了 `float4` 之类的硬件概念，编译器的推导就没有工作对象了——布局被人写死，退化成 Triton 的局面：调度藏在黑盒里，专家拿不到入口。

### host 侧：编译、分配、调用、验证

```python
n, block_size = 1 << 20, 256
kernel = tilelang.compile(vector_add(n, block_size), out_idx=[2], target="cuda")
```

`tilelang.compile` 走完整条五阶段管线：Parser → IR Builder → Optimization → Codegen → 加载。`target="cuda"` 指定生成 CUDA。

`out_idx=[2]` 声明第 2 个参数（C）是输出。TileLang 据此生成 host 侧的 wrapper：调用 `kernel(a, b)` 时在内部为 C 分配显存，算完返回。

```python
a = torch.ones(n, device="cuda", dtype=torch.float32)
b = torch.ones(n, device="cuda", dtype=torch.float32)
c = kernel(a, b)
assert torch.equal(c, 2 * torch.ones(n, device="cuda"))
```

输入用 torch 在 GPU 上分配，省掉了手写 CUDA 里三次 cudaMalloc、两次 cudaMemcpy、cudaDeviceSynchronize 的样板。`kernel(a, b)` 触发编译好的 CUDA kernel 执行。n 和 block_size 在编译期固定为 1048576 和 256，传入的 tensor 长度必须匹配。断言验证全部元素等于 2。

```python
print(kernel.get_kernel_source())
```

打印生成的 CUDA 源码。这是 TileLang 的验证工具：描述和生成物可以逐行对照，下一节做这件事。

## 生成物对照

`get_kernel_source()` 的输出（有省略）：

```
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(
        const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (A[...] + B[...]);
}
```

逐行对回源码和编译器阶段：

- `extern "C" __global__ void main_kernel(...)`：一个 CUDA kernel。`@T.prim_func` 的 `main` 就是它，名字加了 `_kernel` 后缀。
- `__launch_bounds__(256, 1)`：`threads=256` 的落点。它约束编译器按每 block 256 线程生成代码，和 grid 声明一一对应。
- 三个 `const float* __restrict__` 参数：`T.Tensor((n,), "float32")` 的落点。签名里的形状信息在 Optimization 阶段用完了，剩下裸指针交给 Codegen；`__restrict__` 是编译器从"无别名读写"推导后加上的。
- `blockIdx.x * 256`：`bx * block_size` 的落点。bx 就是 blockIdx.x，256 就是 block_size。
- `threadIdx.x`：`T.Parallel(256)` 的落点。迭代变量 i 被编译器推导成 threadIdx.x，一一对应。
- 索引公式 `blockIdx.x * 256 + threadIdx.x`：这就是手写 CUDA 里程序员要自己想出来的公式。这里没人写它，它是 `bx * block_size + i` 三个符号各自替换成硬件量之后自动出现的——符号替换正是"索引到地址的代数映射"的字面含义。

源码里每个构造在生成物里有唯一的落点，没有凭空多出来的部分。反方向也成立：生成物里的每样东西都能指回源码里的一个声明。两个方向都对上，才说明程序写的东西和让渡出去的东西边界清晰。

## 机制回顾

把因果链收拢成一张表：

| DSL 构造 | 给编译器的东西 | 编译器阶段 |
|---|---|---|
| `T.Tensor((n,), "float32")` 静态形状 | 布局代数映射的编译期常数 | Optimization |
| `T.Kernel(grid, threads=...)` | 归属：作用域边界、迭代分配上限、无跨 block 冲突 | Optimization |
| `T.Parallel(...)` | 可迭代空间的并行性 | Optimization |
| 不含硬件概念的数据流代码 | 可变换的 IR 工作对象 | Optimization 全程 |
| `out_idx`、torch tensor | host 侧 wrapper 的形状依据 | Codegen |

设计目标一句话：让程序员写的每一行都对应一个编译器可分析的声明，而不是一堆编译器无法介入的指令。向量加法是这套机制的最简输入，推断没有取舍。
