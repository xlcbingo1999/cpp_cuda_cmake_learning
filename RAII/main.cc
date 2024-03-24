#include "file.h"
#include "mutex.h"
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;
int shared_data = 0;

void increment() {
    for (int i = 0; i < 10000; i++) {
        RAII::LockGuard lock(mtx);
        ++shared_data;

        // 这个作用域结束之后就会调用析构函数把mtx解锁了
    }
}

int main() {
    RAII::File myFile("/home/netlab/xlc_interview/project/example_cmake_engine/RAII/resource/a.txt");
    if (myFile.getHandle().is_open()) {
        std::cout << "File is open" << std::endl;
    } else {
        std::cout << "Failed to open" << std::endl;
    }

    std::thread t1(increment);
    std::thread t2(increment);
    
    t1.join();
    t2.join();
    
    std::cout << "Shared data: " << shared_data << std::endl;
}