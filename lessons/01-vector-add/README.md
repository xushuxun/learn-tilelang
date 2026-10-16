# Vector Add —— 从论文到 TileLang 的第一课

目标：读完这一课，能直接读原论文 [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/abs/2504.17577)（北大、微软研究院）。结构：先总结论文，再用向量加法把论文的概念逐个落地。

## 论文总结

问题。AI kernel 的性能高度依赖手写优化，但手写难。难点不在于数据流——AI kernel 的数据流模式高度雷同，基本都是"在 DRAM 和 SRAM 之间搬 tile，对 tile 做计算"——而在于数据流之外的四项调度工作：线程绑定（tile 操作和数据映射到哪些线程）、内存布局（数据在物理内存里怎么排布以避开 bank conflict、保证合并访存）、张量化（怎么用 Tensor Core 这类硬件指令）、流水线（搬运和计算怎么重叠以藏住访存延迟）。

对现有工具的不满。Triton 这类 DSL 把调度藏在自动策略后面，写简单，但专家想干预时拿不到入口——量化 GEMM 需要内联汇编做类型转换、自定义布局对齐硬件 buffer，在 Triton 里都很别扭。TVM 把计算和调度分开，但调度程序本身难写难懂。

方案。TileLang：数据流和调度解耦。用户用 tile 算子描述数据流；调度由编译器自动推导，推导不够时用户可以拿调度原语覆盖。三条设计原则：Pythonic（嵌在 Python 里）、dataflow-centric、composable（kernel、算子、调度策略可组合）。

核心机制，三个：

- tile 是一等对象：一块有形状的数据，由某个并行执行单元（warp、thread block）拥有和操纵。`T.Kernel` 声明 tile 的归属
- 每个 tile 算子（T.copy、T.gemm、T.reduce、T.atomic、T.Parallel）实现两个接口：Lower（如何降级成线程级代码）和 InferLayout（对 buffer 布局的要求）。解耦靠这两个接口落地
- Layout Inference：编译器按算子优先级推断所有 buffer 的布局和线程绑定，约束最强的算子（如 GEMM）先定，其余适配；Layout 被建模成索引到地址的代数映射，寄存器上的扩展叫 Fragment

编译管线五阶段：Parser（Python AST）→ IR Builder（TVM IR）→ Optimization（Layout Inference、pipeline 推导、尾块切分）→ Codegen（CUDA/HIP 源码）→ 加载执行。

效果。FlashAttention 上比 FlashAttention-3、Triton、PyTorch 分别快 1.36×、1.41×、1.70×；Mamba-2 的 chunk 算子比 Triton 平均快 1.77× 到 2.10×；GEMM 达到 cuBLAS 水平。论文的主张：默认调度就能到 SOTA，专家入口又开着。

## 问题与手写 CUDA

C[i] = A[i] + B[i]，N 个元素互不依赖。手写版 `vector_add.cu` 完整跑通：

```
$ nvcc -O2 -o vector_add vector_add.cu && ./vector_add
vector add ok, n = 1000003, blocks = 3907
```

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;  // 索引公式：thread 坐标 -> 数据坐标
if (i < n)                                       // 边界保护
    c[i] = a[i] + b[i];
```

这段代码混着两类内容：数据流（加什么、写到哪里）和调度（哪个 thread 算哪个元素、地址怎么展开、访存怎么组织）。论文说的"交织"，就是这两类内容写在同一行代码区域里。

## TileLang 编程心智，一步一步

同一个例子，按写代码时脑内思考的顺序走一遍。`vector_add_tilelang.py`：

```
$ python vector_add_tilelang.py
vector add ok, n = 1048576
```

每一步只做一件事。

### 从数据出发，不从线程出发

手写 CUDA 的第一个念头是"要有多少个线程"。TileLang 的第一个念头是"数据长什么样"：

```python
@T.prim_func
def main(
    A: T.Tensor((n,), "float32"),
    B: T.Tensor((n,), "float32"),
    C: T.Tensor((n,), "float32"),
):
```

三个一维 tensor，形状 n，float32。形状和 dtype 是全部类型信息：编译器从这里推导访存的宽度、类型、对齐，不需要指针。这一步只回答"输入输出是什么"，不碰执行。

### 切数据

把 n 个元素切成若干块，每块 block_size 个：

```
n 个元素 → ceil(n / block_size) 个 tile，每 tile block_size 个元素
```

切多大、切几块，是性能决策：tile 大小决定每 block 的工作量，块数决定 grid 规模。这类决策 TileLang 留给程序员，因为它是性能的来源。

### 声明每块数据的归属

```python
with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as bx:
```

`T.Kernel` 声明：有 ceildiv(n, block_size) 个 block，每个 block block_size 个线程；`as bx` 拿到当前 block 的坐标。这一步把第二步切出来的 tile 和硬件执行单元对上号：一个 tile 归一个 block。`with` 块内的代码，描述的就是"一个 block 对自己名下那个 tile 要做什么"。

### 写数据流

tile 内的计算直接写：

```python
C[bx * block_size + i] = A[bx * block_size + i] + B[bx * block_size + i]
```

`bx * block_size` 是 tile 的起始下标，`+ i` 是 tile 内第 i 个元素。读作：block bx 的第 i 次迭代，把 A、B 对应元素相加写进 C。这里没有 threadIdx，没有地址展开——写代码时只想数据怎么走，不想线程怎么走。

### 声明块内并行的方式

```python
for i in T.Parallel(block_size):
```

这个循环有 block_size 次迭代，`T.Parallel` 声明它们并行执行：哪次迭代由哪个线程做，由编译器推导，代码不指定。迭代数等于线程数时，推导结果是一一对应；改成 `T.Parallel(block_size * 4)`，实测编译器会展开成每线程 4 次迭代，并把 4 次 float 访存合并成一条 float4——向量化是推导的副产品。

### 编译，然后验证

`tilelang.compile` 走完编译管线：解析 → TVM IR → Optimization（Layout Inference 等）→ 生成 CUDA 源码 → 加载。生成物可以打印出来：

```
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(
        const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (A[...] + B[...]);
}
```

对照写代码时的五步：tensor 签名变成了三个指针参数（第一步）；切分和归属变成了 grid 维数和 `__launch_bounds__(256, 1)`（第二、三步）；数据流那行还在，坐标展开成 `blockIdx.x * 256 + threadIdx.x`（第四步）；`T.Parallel` 变成了 `threadIdx.x` 一一对应（第五步）。描述里每个构造，在生成物里有唯一的落点。

心智模型就这六步的次序：数据 → 切分 → 归属 → 数据流 → 并行声明 → 验证生成物。前两步是决策，中间两步是描述，第五步是让渡，第六步是确认让渡的结果。

## 背后的机制

第五步让渡出去的映射，靠论文的两套机制落地。

每个 tile 算子（T.copy、T.gemm、T.reduce、T.Parallel）在编译器里实现两个接口：Lower 把算子降级成线程级代码，InferLayout 声明算子对 buffer 布局的要求。`T.Parallel` 的 Lower 负责把循环变成线程绑定加向量化访存；本例 element-wise 操作约束最松，InferLayout 给出直觉的一维分布。

Layout Inference（§4.2）按优先级推断全部 buffer 的布局：约束最强的算子先定（T.gemm 要求 swizzled shared memory 和特定 fragment），共享 buffer 服从最强者。布局被建模成索引到地址的代数映射（Layout），寄存器上的扩展叫 Fragment（§4.1）。向量加法是这套机制的最简输入，推断没有取舍；GEMM 课会见到取舍。

## 读论文的路径

- §1 Introduction：四项调度工作、Triton 的抽象缺口——本课"论文总结"一节
- §2 A TileLang Example：GEMM 完整例子——对照本课的六步，看数据流和调度各多了什么
- §3.1：tile 定义、`T.Kernel`、显式内存分配——本课"切数据""声明归属"两步
- §3.2：Lower 和 InferLayout——本课"背后的机制"一节
- §3.3：调度原语全集——本课"声明块内并行的方式"一步
- §4.1/4.2：Layout 代数、线程绑定推断——本课"背后的机制"一节
- §4.3 张量化、§4.4 流水线：§4.3 讲怎么调 Tensor Core 指令，§4.4 讲流水线怎么自动推导——向量加法没有涉及，读到时回查 §3.2 的算子机制

## 小结

- 论文一句话：数据流和调度解耦，数据流用 tile 算子描述，调度由编译器推导、原语可覆盖
- 本课落地的概念：tile（数据块归属）、grid/block/thread（执行层次）、T.prim_func/T.Kernel/T.Parallel（语言构造）、Lower/InferLayout（算子接口）、Layout Inference（布局推断）
- 验证工具：`kernel.get_kernel_source()`，描述和生成物逐行对照
