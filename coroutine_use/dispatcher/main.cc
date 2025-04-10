#include "executor.h"
#include "utils.h"
#include "task.h"
#include "channel.h"
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

Task<void, LooperExecutor> Producer(Channel<int> &channel) {
    int i = 0;
    while (i < 10) {
        debug("send: ", i);
        
        co_await (channel << i++);
        co_await std::chrono::milliseconds(300); 
    }

    channel.close();
    debug("close channel, exit.")
}

Task<void, LooperExecutor> Consumer(Channel<int> &channel) {
    while (channel.is_active()) {
        try {
            int receive;
            co_await (channel >> receive);
            debug("receive: ", receive);
            co_await std::chrono::milliseconds(2000); 
        } catch (std::exception &e) {
            debug("exception: ", e.what());
        }
    }

    debug("exit consumer")
}

void run_channel() {
    auto channel = Channel<int>(2);
    auto producer = Producer(channel);
    auto consumer = Consumer(channel);

    producer.get_result();
    consumer.get_result();
}

int main() {
    // run_simple_task();
    // run_sleep_executor();

    run_channel();

    return 0;
}