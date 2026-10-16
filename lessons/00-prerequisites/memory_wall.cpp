#include <thread>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// 每个线程对分到的区间求和：纯访存，无计算
void sum_range(const float* a, size_t lo, size_t hi, float* out) {
    float s = 0;
    for (size_t i = lo; i < hi; ++i) s += a[i];
    *out = s;
}

int main(int argc, char** argv) {
    const size_t n = 1ull << 28; // 1 GB float
    float* a = (float*)aligned_alloc(64, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) a[i] = 1.0f;

    printf("%8s %12s %10s\n", "threads", "time_ms", "GB/s");
    for (int t = 1; t <= 64; t *= 2) {
        std::vector<float> part(t);
        std::vector<std::thread> ths;

        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < t; ++k)
            ths.emplace_back(sum_range, a, k * n / t, (k + 1) * n / t, &part[k]);
        for (auto& th : ths) th.join();
        auto ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();

        double gbs = n * sizeof(float) / ms / 1e6;
        printf("%8d %12.1f %10.1f\n", t, ms, gbs);
    }
    free(a);
    return 0;
}
