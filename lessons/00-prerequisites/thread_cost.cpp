#include <pthread.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <string>

// 从 /proc/self/status 里读一项，比如 VmStk（栈占用）
static long read_kb(const char* key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) != 0) continue;
        std::istringstream ss(line);
        std::string name, unit;
        long value;
        ss >> name >> value >> unit;
        return value;
    }
    return -1;
}

int main() {
    // 系统给一个线程准备的栈有多大
    size_t stacksize = 0;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_getstacksize(&attr, &stacksize);
    pthread_attr_destroy(&attr);

    printf("default thread stack : %8zu KB\n", stacksize / 1024);
    printf("VmSize (before)      : %8ld KB\n", read_kb("VmSize:"));

    std::thread th([]{ while (true) std::this_thread::sleep_for(std::chrono::seconds(1)); });

    printf("VmSize (after)       : %8ld KB\n", read_kb("VmSize:"));
    printf("VmRSS  (after)       : %8ld KB\n", read_kb("VmRSS:"));

    th.detach();
    return 0;
}
