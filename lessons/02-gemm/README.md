# Tilelang 实现 GEMM

GEMM：C = A @ B。向量加法之后第一个有数据复用的算子：C 的每个元素要读 A 的一行和 B 的一列，不做复用时计算被访存拖死。这一课用六步心智模型写完它，再读生成物，看编译器从描述里推导出了什么。

## 例子

`gemm.py`：

```
$ python gemm.py
gemm ok: rel_err = 7.75e-04
time: 1.753 ms, 9.80 TFLOPS
```

2048×2048×2048，校验用相对误差（L2 范数比），原因见"张量化"一节。

## 数据

三个二维 tensor：A (M, K)、B (K, N)、C (M, N)。签名给出形状和 dtype，编译器从这里推导全部访存类型。

## 切数据

两个层次的切分。

C 的切分：C 是 M×N，切成 (M/block_M) × (N/block_N) 个 tile，每 tile block_M×block_N。每个 tile 由一个 block 负责——这是 block 级并行。

K 的切分：累加维 K 不切就没有复用。切成 K/block_K 个片，每片 block_K 宽。block 一次只搬一个 K 片进片上存储，对这一片做矩阵乘并累加，然后换下一篇。A 的一个 block_M×block_K 片被这一列的复用、B 的 block_K×block_N 片被这一行的复用，都在 shared memory 里完成。

切分参数 block_M=128、block_N=128、block_K=32 是程序员的决策，直接决定生成物的形态和性能。

## 归属与显式内存放置

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), "float32")
    B_shared = T.alloc_shared((block_K, block_N), "float32")
    C_local  = T.alloc_fragment((block_M, block_N), "float32")
```

`T.Kernel` 声明二维 grid：block (bx, by) 负责 C 的第 (by, bx) 个 tile。`T.alloc_shared` 和 `T.alloc_fragment` 是显式内存放置：A/B 的当前 K 片放 shared memory（block 内共享、延迟低），累加器 C_local 放寄存器 fragment（每个 thread 持有自己的一份切片）。这是 TileLang 对 Triton 的关键区别：buffer 在硬件存储层次的哪一层，程序员声明，编译器不猜。

## 数据流

```python
T.clear(C_local)
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
T.copy(C_local, C[by * block_M, bx * block_N])
```

五行数据流，三个 tile 算子。`T.copy` 搬运（global → shared、shared → global）；`T.gemm` 在片上对两个 shared 片做矩阵乘、结果累加进 fragment；`T.clear` 初始化累加器。整个 K 循环是"搬一片、算一片"，循环体里没有线程、没有地址展开、没有同步。

## 调度：流水线

`T.Pipelined(num_stages=3)` 声明 K 循环按 3 级流水执行：搬运第 ko+2 片和计算第 ko 片重叠。没有这一行，循环就是搬完再算、算完再搬，SM 在搬运期间空转。num_stages 是程序员的调度决策，内存代价是三份 shared buffer（生成物里能看到 `(ko + 2) % 3` 的轮转）。

## 读生成物

描述五行，生成物 137 行 CUDA（`kernel.get_kernel_source()`）。挑四处读：

搬运是异步的，且做了向量化：

```cpp
tl::cp_async_gs<16>(...);   // 全局 -> shared 的异步拷贝，每次 16 字节
tl::cp_async_commit();
tl::cp_async_wait<2>();     // 等最旧的一级流水
__syncthreads();
```

`cp_async` 不经过寄存器、搬运和计算可以重叠，`num_stages=3` 变成三片 buffer 的轮转加上 wait/commit 的配对同步——这些都没写在描述里，是 T.copy 和 T.Pipelined 的 Lower 规则生成的。

shared memory 的地址里有大量位运算：

```cpp
((float*)A_shared)[(i * 512) + ((threadIdx.x >> 3) * 32) + (... & 1) * 16) + (...)]
```

这是 swizzled layout：写入时把地址按位打散，让同一轮访存的线程落在不同的 memory bank 上，避免 bank conflict。InferLayout 为 T.gemm 推导的，程序员没有指定任何布局。

计算走的是 Tensor Core：

```cpp
tl::mma_sync<tl::DataType::kTensorFloat32, ..., 16, 8, 8, ...>(...)
```

`kTensorFloat32` 即 TF32 指令，16×8×8 是 mma 形状，A_local/B_local 是从 shared 搬进寄存器的 fragment。fp32 输入走 TF32 意味着尾数只有 10 位——这就是校验用相对误差的原因，也是 9.80 TFLOPS 的来源：CUDA core 的 fp32 FMA 在这张卡上约 15 TFLOPS，TF32 tensor core 的峰值高一个数量级，这里 128×128×32 的切分还没吃满。

## 调参就是改切分

改 block_M/block_N/block_K/num_stages 这四个数，重新 compile，生成物跟着变：block_K 减半则 cp_async 的宽度减半；num_stages 加一则轮转模数加一、shared memory 占用涨一份。描述不变长，生成物变——这一课开头的切分决策，到这里变成可以实测的搜索空间。

### 一次实测的调参过程

对 2048³ 搜索 7 组参数，每组三步：改参数重新编译、CUDA event 计时（预热 3 次、测 10 次取均值）、失败的直接排除：

```
block=(128,128,32) stages=2: 1.748 ms  9.83 TFLOPS
block=(128,128,32) stages=3: 1.762 ms  9.75 TFLOPS
block=(128,128,32) stages=4: FAIL（shared memory 超限）
block=(128, 64,32) stages=3: 1.841 ms  9.33 TFLOPS
block=( 64, 64,32) stages=3: 11.411 ms  1.51 TFLOPS
block=(128,128,64) stages=3: FAIL（shared memory 超限）
block=( 64, 64,64) stages=4: FAIL（shared memory 超限）
```

约束关系在数据里可见：block_M×block_N 决定累加器大小——128×128 fp32 是 64 KB 寄存器，128 线程摊到每线程 128 个，接近上限；block_K×num_stages 决定 shared memory 占用——stages=4 时三份 A/B buffer 超过这块卡 99 KB 的动态 shared memory 上限，编译或 launch 报错。两项相加卡死可行域。缩小 tile（64×64）两项都省，但每个 block 的并行度不够，性能掉一个数量级——搜索不是单调的。

单次迭代几秒：改数字、重编译、计时。同样的过程在手写 CUDA 里是重写 kernel 再 nvcc，以小时计。这是描述即 IR 的直接收益。

## 调优的三个层次

手工调参循环再往上，是两级自动化。三级的结构相同，搜索对象逐级放大：

| 层次 | 搜索对象 | 谁执行 | 正确性保障 |
| --- | --- | --- | --- |
| 手工调参 | block_M/N/K、num_stages 这几个数字 | 人 | 人写断言 |
| TileLang autotuner | 同一套数字的配置空间 | 编译器框架 | 内置校验，并行编译候选 |
| TileFoundry | 整个程序文本 | agent | describe-check-optimize 循环 + L1–L3 分级验证 |

第一级就是上一节的实测。第二级，TileLang 自带 [autotuner](https://tilelang.com/programming_guides/autotuning.html)：把配置空间声明出来，框架并行编译候选 kernel、逐个计时、用参考输出自动校验——手工循环的三个步骤原样保留，只是由框架驱动。

第三级是 tile-ai 团队的 [TileFoundry](https://github.com/tile-ai/TileFoundry)：agentic 的程序生成平台。它解决的问题和 TileLang 同构，但让渡出去的东西多了一层。对照看：

- TileLang 让渡线程代码，保留切分决策；手工调参是人来做这个决策
- TileFoundry 让渡程序本身，保留验证：agent 在 HIR（高级 IR）层描述模型、写优化版本、用 `tilefoundry check` 和参考实现对拍，框架的调度设施（多面体分析、storage-driven 指令选择、带空洞的 kernel 骨架）把"试什么"变成可计算的查询，L1（逐层）/L2（整链）/L3（逐 token 复现官方输出）三级信任标注每个程序验证到什么程度
- 它的设计原则"fail closed, verify explicitly"——check 没有默认容差，误差界由调用方声明——和手工调参里"失败的配置直接排除"是同一个纪律：静默通过不产出任何知识

交叉验证一个判断：三个层次共用的基本单元相同——一次候选程序 + 一次计时 + 一次显式正确性判定。自动化的收益来自把这个单元跑得更多、更广，而不是来自替换它。手工调参一轮几秒、autotuner 一轮并行几百个配置、TileFoundry 一轮是 agent 的一次工具调用（实测记录：Claude Opus 5 自主工作 2.1 小时、177 次工具调用产出 Qwen3-1.7B 的 TileLang 解码实现，单 H200 实测 612 tok/s）。

## 小结

- GEMM 的心智模型：C 切 tile 归 block，K 切片进 shared memory，数据流是"搬一片算一片"
- 三个新语言构造：`T.alloc_shared`/`T.alloc_fragment`（显式内存放置）、`T.copy`（搬运算子）、`T.gemm`（片上矩阵乘）、`T.Pipelined`（流水调度）
- 生成物里编译器推导的：cp_async 异步搬运与同步配对、swizzled layout、TF32 mma 指令
- 性能决策留在程序员手里：切分参数和流水级数；描述五行，生成物 137 行
- 调优三级：手工调参搜数字、autotuner 搜配置空间、TileFoundry 让 agent 搜程序文本；基本单元不变——候选 + 计时 + 显式判定
