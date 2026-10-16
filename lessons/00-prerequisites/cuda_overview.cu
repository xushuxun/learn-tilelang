#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; } } while (0)

// 驱动控制类操作：查硬件参数、分配显存、搬数据
int driver_ops() {
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("device            : %s\n", prop.name);
    printf("SM count          : %d\n", prop.multiProcessorCount);
    printf("global memory     : %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

    // 实测 H2D 带宽：CPU 内存 -> GPU 显存，走 PCIe
    const size_t bytes = 256ull << 20;
    void *h, *d;
    CHECK(cudaMallocHost(&h, bytes));
    CHECK(cudaMalloc(&d, bytes));
    memset(h, 1, bytes);

    CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); // 预热
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i)
        CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / 10;

    printf("H2D bandwidth     : %.1f GB/s (%zu MB in %.2f ms)\n",
           bytes / ms / 1e6, bytes >> 20, ms);

    CHECK(cudaFreeHost(h));
    CHECK(cudaFree(d));
    return 0;
}

// 动态统计：每个线程记录自己跑在哪个 SM、执行了多少周期
__global__ void probe(const float* buf, int* smid, long long* cycles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int s;
    asm("mov.u32 %0, %%smid;" : "=r"(s));

    long long t0 = clock64();
    // 一段可测量的工作：多次读显存，读到的值真的参与累加
    float sum = 0;
    for (int i = 0; i < 100; ++i)
        sum += buf[(idx + i * 32) % 4096];
    long long t1 = clock64();

    smid[idx] = (int)s + (sum == -1.0f ? 1 : 0); // sum 被使用，循环不可被优化掉
    cycles[idx] = t1 - t0;
}

int main() {
    if (driver_ops()) return 1;

    const int n_blocks = 8, n_threads = 128, n = n_blocks * n_threads;
    float* buf;
    int* smid;
    long long* cycles;
    CHECK(cudaMallocManaged(&buf, 4096 * sizeof(float)));
    CHECK(cudaMallocManaged(&smid, n * sizeof(int)));
    CHECK(cudaMallocManaged(&cycles, n * sizeof(long long)));

    probe<<<n_blocks, n_threads>>>(buf, smid, cycles);
    CHECK(cudaDeviceSynchronize());

    printf("tid smid cycles\n");
    for (int i = 0; i < n; i += 37)  // 抽样打印，避免刷屏
        printf("%3d %4d %6lld\n", i, smid[i], cycles[i]);

    CHECK(cudaFree(buf));
    CHECK(cudaFree(smid));
    CHECK(cudaFree(cycles));
    return 0;
}
