#include "executor.h"
#include "utils.h"
#include "task.h"
#include "executor.h"

Task<int, AsyncExecutor> simple_sub_task1() {
    debug("sub_task1 start ...");
    // std::this_thread::sleep_for(std::chrono::seconds(1)) ; // 这个是纯阻塞IO
    co_await std::chrono::seconds(1); // 现在就可以让sleep_awaiter接管，然后由sleep_awaiter中的调度器协助调度
    debug("sub_task1 returns after 1s.");
    co_return 2;
}

Task<int, AsyncExecutor> simple_sub_task2() {
    debug("sub_task2 start ...");
    // std::this_thread::sleep_for(std::chrono::seconds(2)) ; // 这个是纯阻塞IO
    co_await std::chrono::seconds(2); // 现在就可以让sleep_awaiter接管，然后由sleep_awaiter中的调度器协助调度
    debug("sub_task2 returns after 2s.");
    co_return 3;
}

Task<int, SharedLooperExecutor> simple_task() {
    debug("task start ...");
    co_await std::chrono::milliseconds(100); 
    debug("after 100ms ...");
    auto result1 = co_await simple_sub_task1();
    debug("task from sub_task1: ", result1);

    co_await std::chrono::milliseconds(200); 
    debug("after 200ms ...");
    auto result2 = co_await simple_sub_task2();
    debug("task from sub_task2: ", result2);
    
    co_return 1 + result1 + result2;
}

void run_simple_task() {
    auto simpleTask = simple_task();

    // 异步
    simpleTask.then([](int i) {
        debug("simple task end: ", i); // 这个是异步的，因此会由coroutine所在的thread打印处理
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
}

void run_sleep_executor() {
    auto executor = SleepExecutor();
    debug("start");

    executor.execute([]() { debug(2); }, 100);
    executor.execute([]() { debug(1); }, 50);
    executor.execute([]() { debug(6); }, 1000);
    executor.execute([]() { debug(5); }, 500);
    executor.execute([]() { debug(3); }, 200);
    executor.execute([]() { debug(4); }, 300);
    
    executor.shutdown();
    executor.join();
}

int main() {
    run_simple_task();
    run_sleep_executor();

    return 0;
}