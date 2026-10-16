# 课程 01：Vector Add —— CUDA C++ 与 TileLang

## 这节课要解决的问题

- 为什么第一个算子选向量加法：它足够简单，却能暴露 kernel 编程的全部骨架
- 同一道题用 CUDA C++ 写一遍、再用 TileLang 写一遍，看清两者在"写什么"上的差别

## 向量加法在 GPU 上的图景

- 数据并行的本质：每个输出元素互不依赖
- grid / block / thread 三级组织如何映射到 N 个元素
- 需要多少线程、线程超出长度怎么办（边界保护）

## CUDA C++ 手写版本

- kernel 函数：`__global__`、索引计算、越界判断
- 主机端流程：内存分配、Host↔Device 数据拷贝、launch 配置、同步与计时
- 编译与运行：`nvcc` 与 `cuda-python` 两种上手方式
- 常见错误：忘记同步、索引算错一位、grid 尺寸取整

## TileLang 版本

- 用 T.Kernel 描述同样的计算：placeholders、computations、thread binding
- TileLang 在背后做了什么：索引生成、边界处理、launch 参数
- 编译产物：生成的 CUDA 源码长什么样，和手写版逐行对照

## 性能与正确性验证

- 用 numpy 做 reference，校验两种实现的输出
- nvprof / nsys 看运行时间，对比手写与 TileLang 版本的差距
- 改变 N 与 block 大小，观察性能曲线的变化

## 小结与下一课预告

- 向量加法教会我们的最小 kernel 骨架
- 下一课 GEMM：当输出元素之间开始共享输入数据， tiling 为什么成为刚需
