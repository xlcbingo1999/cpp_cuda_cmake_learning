#include "utils.h"
#include "task.h"
#include <thread>

Task<int> simple_sub_task1() {
    debug("sub_task1 start ...");
    std::this_thread::sleep_for(std::chrono::seconds(1)) ;
    debug("sub_task1 returns after 1s.");
    co_return 2;
}

Task<int> simple_sub_task2() {
    debug("sub_task2 start ...");
    std::this_thread::sleep_for(std::chrono::seconds(2)) ;
    debug("sub_task2 returns after 2s.");
    co_return 3;
}

Task<int> simple_task() {
    debug("task start ...");
    auto result1 = co_await simple_sub_task1();
    debug("task from sub_task1: ", result1);
    auto result2 = co_await simple_sub_task2();
    debug("task from sub_task2: ", result2);
    
    co_return 1 + result1 + result2;
}

int main() {
    auto simpleTask = simple_task();

    // 异步
    simpleTask.then([](int i) {
        debug("simple task end: ", i);
    }).catching([](std::exception &e) {
        debug("error occurred", e.what());
    });

    // 同步
    try {
        auto i = simpleTask.get_result();
        debug("simple task end from get: ", i);
    } catch (std::exception &e) {
        debug("error: ", e.what());
    }

    return 0;
}