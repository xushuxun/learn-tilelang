# 前置知识 —— 从 CPU 线程到 GPU 硬件

## 这节课要解决的问题

- 面向读者：深刻理解C++/系统线程，但是没接触过GPU编程。建立CUDA编程心智模型

## CPU每个线程的成本，多线程内存墙

一个线程在硬件上占着什么？`thread_cost.cpp`：

```
$ g++ -O2 -pthread -o thread_cost thread_cost.cpp && ./thread_cost
default thread stack :     8192 KB
VmSize (before)      :     6376 KB
VmSize (after)       :    14572 KB
VmRSS  (after)       :     3844 KB
```

一个线程：8 MB 栈预留，外加内核任务结构。64 GB 机器只够八千个。切换线程还会弄丢 cache 里上一任的工作集。CPU 线程是稀缺资源。

那几十个线程一起跑，能有多快？`memory_wall.cpp` 让 1 到 64 个线程各扫一遍 1 GB 数组：

```
$ g++ -O2 -pthread -o memory_wall memory_wall.cpp && ./memory_wall
 threads      time_ms       GB/s
       1        150.2        7.1
       2         76.5       14.0
       4         41.8       25.7
       8         25.4       42.3
      16         20.9       51.3
      32         20.3       52.8
      64         22.1       48.5
```

16 线程之前，核数翻倍，带宽翻倍；16 之后，带宽停在 52 GB/s 不再增长——内存总线饱和，再加线程只是排队。

两个程序各说明一件事：

- `thread_cost.cpp`：线程数量有上限
- `memory_wall.cpp`：带宽有上限
