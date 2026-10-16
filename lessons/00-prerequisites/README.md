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


## 了解 PCIe

先分清两种访问。CPU 访问自己的内存：一条 `mov` 指令，load/store 直接落到内存控制器。纳秒级。

GPU 是外设，显存不在 CPU 的物理地址空间里，CPU 不能 load/store 显存。CPU 和外设之间是两条路：

- 编程外设寄存器：驱动把外设寄存器映射到地址空间（MMIO），CPU 读写这些地址，实际走 PCIe 上的事务包
- 搬运数据：CPU 在内存备好数据，写外设的 DMA 描述符，外设自己经 PCIe 把数据读过去

两条路都受 PCIe 约束：一次事务有固定开销，带宽有上限。这里展开"总线"是什么。

总线是设备之间共享的通信通路，由三部分组成：数据线（传什么）、地址线（传给谁）、控制线（读还是写）。一次传输是一个事务：主设备发出地址和命令，从设备把数据放上总线。同一时刻一条总线只容得下一个事务，所以总线的两个指标是延迟和带宽。

机器里不止一条总线：CPU 和内存之间是内存总线，CPU 和外设之间是 I/O 总线。CPU 访问内存快，因为内存总线短、专用；访问外设慢，因为要穿过 I/O 总线这层共享介质。

Linux 把所有总线上的设备挂在同一棵树里，lspci 枚举这棵树：

```
$ lspci
3b86:00:00.0 System peripheral: Red Hat, Inc. Virtio file system (rev 01)
5582:00:00.0 SCSI storage controller: Red Hat, Inc. Virtio 1.0 console (rev 01)
6e29:00:00.0 3D controller: Microsoft Corporation Basic Render Driver
a380:00:00.0 3D controller: Microsoft Corporation Device 008a
ee7e:00:00.0 3D controller: Microsoft Corporation Basic Render Driver
$ lspci -t
-[3b86:00]---00.0
-[5582:00]---00.0
-[a380:00]---00.0
-[6e29:00]---00.0
-[ee7e:00]---00.0
```

编号 `a380:00:00.0` 是设备在树上的坐标：总线号、设备号、功能号。lspci -t 显示拓扑：每个方括号是一条总线，每个 `---` 是挂在它下面的设备。

历史背景：早期 PCI 是共享并行总线，所有设备挤在同一组线上，时钟频率做不高。PCIe 改成点对点串行链路，每对设备之间独立传包，名字里的"总线"是历史遗留。CPU 一侧的扇出点叫 Root Complex，设备挂在树上，地址按树分配。设备之间以事务包（TLP）通信：一次内存读写被封装成一个包，加上包头开销后经链路发出。

每条链路的带宽由代际和通道数决定：gen4 x16 约 32 GB/s，gen5 x16 约 64 GB/s；笔记本常见的 gen4 x8 约 16 GB/s。所有跨 PCIe 的传输都以这个带宽为上限。`lspci -v` 可以看单个设备和它绑定的驱动：

```
$ lspci -v -s a380:00:00.0
a380:00:00.0 3D controller: Microsoft Corporation Device 008a
        Flags: bus master, fast devsel, latency 0, NUMA node 0
        Kernel driver in use: dxgkrnl
        Kernel modules: dxgkrnl
```

`Kernel driver in use` 是绑在这个设备上的内核驱动。本机是 WSL2，GPU 以半虚拟化设备出现（设备号 008a），绑的是微软的 dxgkrnl 而不是 nvidia；原生 Linux 上这里会显示 nvidia，固件在 `/lib/firmware/nvidia/<驱动版本>/`。

## Linux 内核视角下的 GPU 

对内核来说，GPU 和其他 PCIe 设备没有区别，走同一套驱动模型。模型里三样东西：总线、设备、驱动。驱动就是一个内核模块：模块加载时调用 `pci_driver_register()` 把自己挂到 PCI 总线的驱动链表上，设备枚举时按设备号匹配，匹配成功就调驱动的 probe 回调。模块加载和设备绑定是同一件事的两面。

```
$ ls /sys/bus/pci/devices/a380:00:00.0/
...  class  config  driver  irq  resource  vendor ...
```

- `config`：配置空间，PCIe 标准头部（厂商、设备号、BAR），驱动据此识别设备
- `resource`：设备申请的地址范围，驱动据此把显存和寄存器映射进内核地址空间
- `driver`：指向绑定驱动的符号链接，"设备 ↔ 驱动"关系的落点

probe 回调里驱动做三件事：

- 映射资源：把显存 BAR 映射进内核地址空间，读 BAR0 就能访问显存
- 加载固件：NVIDIA 这一代 GPU 带 GSP（GPU System Processor），固件在 `/lib/firmware/nvidia/<驱动版本>/`，驱动经 PCIe 写进卡里，负责设备初始化和电源管理
- 注册接口：创建 `/dev` 下的字符设备，用户程序 open + ioctl 访问 GPU

### 找到 nvidia 驱动文件

驱动作为模块存在 `/lib/modules/<内核版本>/` 下，kmod 工具集负责管理这些文件：lsmod 列出已加载模块，modinfo 打印模块元信息，modprobe 加载。用它们看本机的 GPU 驱动：

```
$ lsmod | grep dxgkrnl
dxgkrnl
$ modinfo dxgkrnl
name:           dxgkrnl
filename:       (builtin)
version:        2.0.3
description:    Microsoft Dxgkrnl virtual compute device Driver
license:        GPL
file:           drivers/hv/dxgkrnl/dxgkrnl
```

`filename: (builtin)` 表示这个模块编进了内核镜像，不是一个独立的 .ko 文件。`/lib/modules/<内核版本>/` 下有一份清单：

```
$ ls /lib/modules/6.18.54.1-microsoft-standard-WSL2/
build  kernel  modules.builtin  modules.dep  modules.order ...
```

- `modules.builtin`：编进内核的模块名单
- `modules.dep`：可加载模块的依赖关系，modprobe 靠它解析依赖
- `kernel/` 子树：可加载的 .ko 文件

原生 Linux 上 nvidia 是可加载模块，`modinfo nvidia` 的 filename 会指向具体路径，一般在 `/lib/modules/<内核版本>/kernel/drivers/video/nvidia.ko`。

## GPU 硬件特点 与 CUDA 编程模型

GPU 的硬件：几十个 SM（Streaming Multiprocessor，流式多处理器），每个 SM 有自己的寄存器堆、shared memory、调度器；全卡共享显存。SM 是执行单元。

这是理解 GPU 的核心机制：GPU 没有 cache 可以把工作集留住，线程的每次访存都要去显存，延迟数百个时钟周期，单个线程永远在等待。GPU 的解法是数量——一个 SM 驻留上千个线程，某个线程的访存没返回，调度器立刻切到另一个 warp 执行。等待被其他线程的执行填满，这就是"靠调度器在线程间切换来隐藏访存延迟"。GPU 编程的所有优化，最终都落在一件事上：让 SM 上始终有就绪的线程可以切。

warp 和调度器是这套机制的两个部件。

warp：SM 的调度单位。block 被分配到 SM 后，内部线程按 32 个一组切成 warp；一个 warp 内的线程执行同一条指令，只是操作各自的寄存器和数据（SIMT）。代码里的 if 让 32 个线程走上不同分支时，warp 不能拆散：各分支串行执行，不在这条分支上的线程被屏蔽，这叫分支发散。

调度器：SM 上做切换决策的部件。现代架构一个 SM 有多个调度器（Ada 是 4 个），每个掌管自己的 warp 池。每个时钟周期，调度器从就绪的 warp 里选一个发射指令。warp 只有就绪和等待两种状态。切换没有保存恢复的成本——所有 warp 的上下文常驻在寄存器堆里，选中即执行。

把两条合起来是定量的：显存延迟约 400 周期，一个 warp 发出访存指令后要等待约 400 周期，期间需要其他 warp 填上这些周期。SM 驻留上限（本机 1536 线程 = 48 个 warp）就是按这个公式定的。

CUDA 是给这套硬件的编程模型。软件里的每个概念都能在硬件上找到落点，反过来说，硬件的每个部件都通过软件概念暴露给程序员：

| 软件概念 | 硬件落点 |
| --- | --- |
| kernel | 发射到 SM 上执行的一批指令 |
| grid | 一次 launch 的全部 block |
| block | SM 上的调度单位，共享这块 SM 的 shared memory |
| warp | 无软件对应，硬件调度单位，通过分支发散暴露给程序员 |
| thread | 一组寄存器组成的上下文，不占内存 |
| shared memory | SM 片上的存储，延迟远低于显存 |
| global memory | 显存 |

两点补充。block 分配到哪个 SM 由硬件和驱动决定，程序不可指定；block 占住 SM 直到全部线程退出，然后 SM 收回调度下一个 block。

CUDA 本身由两部分组成：nvcc 编译器，把 C++ 代码编译成 GPU 指令；运行时库 libcudart，封装对驱动的调用。程序从 CPU 启动，kernel 调用交给驱动发射到 GPU。

`cuda_overview.cu` 演示这些概念。先看 kernel，逐行注释对应前面讲的硬件：

```cpp
// kernel：__global__ 表示这个函数在 GPU 上执行，由 CPU 端 launch
__global__ void probe(const float* buf, int* smid, long long* cycles) {
    // blockIdx / threadIdx / blockDim：thread 运行时的坐标（软件概念）
    // 每个 thread 执行同一份代码，靠 idx 区分各自处理哪份数据
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // %smid 是硬件寄存器：这个 thread 的 warp 当前被调度在哪个 SM 上
    // 软件层无法指定 SM，只能事后读取——"编号由软件定义、位置由硬件决定"
    unsigned int s;
    asm("mov.u32 %0, %%smid;" : "=r"(s));

    // clock64() 是 SM 的时钟计数器，读两次得到一个周期差
    long long t0 = clock64();

    // 一段可测量的工作：100 次读显存（global memory）
    // buf 的数据真的参与累加，编译器不能把循环优化掉
    // 读显存延迟约 400 周期，单看一个 thread，这段时间全在等待
    float sum = 0;
    for (int i = 0; i < 100; ++i)
        sum += buf[(idx + i * 32) % 4096];

    long long t1 = clock64();

    // 每个 thread 把硬件统计结果写回数组：自己跑在哪个 SM、花了多少周期
    smid[idx] = (int)s + (sum == -1.0f ? 1 : 0);
    cycles[idx] = t1 - t0;
}
```

launch 在 CPU 侧：`probe<<<8, 128>>>` 一次 launch 一个 grid，含 8 个 block、每 block 128 个 thread，共 1024 个 thread；`cudaDeviceSynchronize()` 等 GPU 全部跑完再读结果。运行：

```
$ nvcc -O2 -o cuda_overview cuda_overview.cu && ./cuda_overview
device            : NVIDIA GeForce RTX 4060 Laptop GPU
SM count          : 24
global memory     : 8.6 GB
max threads per SM: 1536
H2D bandwidth     : 11.9 GB/s (256 MB in 22.55 ms)
tid smid cycles
  0    0  27027
 37    0  27045
148    2  25677
259    4  22529
370    4  22583
407    6  26347
518    8  24896
629    8  25115
740   10  23752
851   12  25120
962   14  25728
```

输出回答三个问题：

- smid 列：8 个 block 分散在 SM 0 到 14 上，每个 block 独占一个 SM。分配是运行时决定的，代码里没有写任何相关内容
- cycles 列：每个 thread 做同样 100 次读取，周期数在 22529 到 27080 之间浮动。来源是访存延迟的随机性和同 SM 上其他 warp 的竞争——1024 个 thread 在 SM 上轮换执行，单个 thread 的周期数包含了自己不执行、让位给别人的时间
- tid 只是抽样序号，行与行之间没有执行先后的含义

这就是 profiling 的基本心智：性能不是"程序跑了多久"一个数，而是"每个执行单元在哪、各自花了多少周期"的分布。只看 wall time 会漏掉这个分布里的信息；后面每一课优化 kernel 时，手段都是把统计口径从整个程序拆小——拆到 block、到 warp、到一条访存——找到周期花在了哪里。%smid 和 clock64 是最初级的内置统计工具，专门的 profiler（nsys、ncu）做的是同一件事，只是统计面更全。
