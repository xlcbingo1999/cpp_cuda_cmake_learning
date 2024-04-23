#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

std::atomic_int acnt;

/*
void volatile_test() {
    volatile int i = 200;
    int a = i;
    std::cout << "i: " << a << std::endl;

    // 内联汇编代码
    __asm(
        "mov dword ptr [%ebp-4], 20h"
    );

    int b = i;
    std::cout << "i: " << b << std::endl;
}
*/

void f() {
    // volatile每次都从内存中读取对应的值, 避免编译器不知道的__asm进行修改, 但是编译器优化之后从缓存中读取值, 导致无法读取真正的值
    for (volatile int i = 0; i < 10000; i = i+1) {
        ++acnt; // 原子增加
    }
}

int main() {
    {
        // 创建一个线程池
        std::vector<std::thread> pool;
        for (int i = 0; i < 10; i++) {
            pool.emplace_back(f); 
        }

        for (int i = 0; i < 10; i++) {
            pool[i].join();
        }
    }
    std::cout << "acnt: " << acnt << std::endl;
}