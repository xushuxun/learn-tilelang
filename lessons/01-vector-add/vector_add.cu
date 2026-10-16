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

    // device 侧分配（三次！）
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
