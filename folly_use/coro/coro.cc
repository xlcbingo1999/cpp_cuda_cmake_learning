#include "folly/executors/CPUThreadPoolExecutor.h"
#include <coroutine>
#include <exception>
#include <vector>
#include <map>
// #include <string>

#include <iostream>
#include <folly/init/Init.h>
#include <folly/logging/LogLevel.h>
#include <glog/logging.h>
#include<folly/experimental/coro/Task.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/executors/GlobalExecutor.h>
// #include <sys/select.h>
#include <folly/experimental/coro/SharedMutex.h>
// #include "folly/executors/CPUThreadPoolExecutor.h"


#include <chrono>
#include <iomanip>
#include <thread>

inline char separator() {
    #ifdef _WIN32
      return '\\';
    #else
      return '/';
    #endif
}

const char* file_name(const char *path) {
    const char *file = path;
    while (*path) {
        if (*path++ == separator()) {
            file = path;
        }
    }
    return file;
}


void PrintTime() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;    

    std::cout << std::put_time(std::localtime(&in_time_t), "%T") 
        << "." << std::setfill('0') << std::setw(3) << ms.count();
}


void PrintThread() {
    std::cout << " [Thread-" << std::this_thread::get_id() << "] ";
}

template <typename ...U>
void Println(U... u) {
    int i = 0;
    auto printer = [&i]<typename Arg>(Arg arg) {
        if (sizeof...(u) == ++i) {
            std::cout << arg << std::endl;
        } else {
            std::cout << arg << " ";
        }
    };
    (printer(u), ...);

    std::cout.flush();
}

#define debug(...) \
    PrintTime();   \
    PrintThread(); \
    printf("(%s:%d) %s: ", file_name(__FILE__), __LINE__, __func__); \
    Println(__VA_ARGS__);



std::map<std::string, int> global_value = {
    {"a", 0},
    {"b", 0},
};
int global_count = 0;
folly::coro::SharedMutexFair coro_lock;

static folly::CPUThreadPoolExecutor& get_executor_1() {
    static folly::CPUThreadPoolExecutor executor(2);
    return executor;
}

static folly::CPUThreadPoolExecutor& get_executor_2() {
    static folly::CPUThreadPoolExecutor executor(2);
    return executor;
}

folly::coro::Task<void> a() {
    debug("task a begin sleep 1");
    sleep(1);
    global_value["a"] = ++global_count;
    debug("task a ok ");
    co_return;
}

folly::coro::Task<void> b() {
    debug("task b begin sleep 2");
    sleep(2);
    global_value["b"] = ++global_count;
    debug("task b ok ");
    co_return;
}

folly::coro::Task<bool> MySyncTask() {
    std::vector<folly::coro::Task<void>> sum;
    debug("Coroutine started");

    sum.push_back(a());
    debug("Coroutine triggled a");
    sum.push_back(b());
    debug("Coroutine triggled b");
    
    try {
        co_await folly::coro::collectAllRange(std::move(sum));
        debug("Coroutines get the result of sum");
    } catch (std::exception &e) {
        debug("error: ", e.what());
        co_return false;
    }
    debug("sync task finished");
    co_return true;
}

folly::coro::Task<bool> MySyncTaskV2() {
    std::vector<folly::coro::Task<void>> sum;
    debug("Coroutine started");

    sum.push_back(a());
    debug("Coroutine triggled a");
    sum.push_back(b());
    debug("Coroutine triggled b");

    // 这里会将内部的coroutine丢给executor执行，此时会造成切换thread进行
    co_await folly::coro::collectAllRange(std::move(sum)).scheduleOn(&get_executor_1());
    debug("Coroutines get the result of sum");

    debug("sync task finished");
    co_return true;
}

folly::coro::Task<bool> MyAsyncTask() {
    std::vector<folly::coro::TaskWithExecutor<void>> sum; // 这个模式和我们自己实现的task是保持一致的
    debug("Coroutine started");
    
    sum.push_back(a().scheduleOn(folly::getGlobalCPUExecutor()));
    debug("Coroutine triggled a");
    sum.push_back(b().scheduleOn(folly::getGlobalCPUExecutor()));
    debug("Coroutine triggled b");

    co_await folly::coro::collectAllRange(std::move(sum));
    debug("Coroutines get the result of sum");

    debug("async task finished");
    co_return true;
}

folly::coro::Task<void> getLock1() {
    auto lock = co_await coro_lock.co_scoped_lock();
    debug("getLock1");
    co_return ;
}

folly::coro::Task<void> getLock2() {
    auto lock = co_await coro_lock.co_scoped_lock();
    co_await getLock1().scheduleOn(folly::getGlobalCPUExecutor());
    debug("getLock2");
    co_return ;
}

folly::coro::Task<void> getLock3() {
    co_await getLock1().scheduleOn(folly::getGlobalCPUExecutor());
    debug("getLock3");
    co_return ;
}

int main(int argc, char *argv[]) {
    folly::init(&argc, &argv);

    debug("coro test");

    auto task1 = MySyncTask();
    folly::coro::blockingWait(std::move(task1).scheduleOn(&get_executor_2()));
    debug("a = ", global_value["a"], " b = ", global_value["b"]);
    debug("global_count = ", global_count); 

    printf("\n\n\n");
    
    auto task2 = MySyncTaskV2();
    folly::coro::blockingWait(std::move(task2).scheduleOn(&get_executor_2()));
    debug("a = ", global_value["a"], " b = ", global_value["b"]);
    debug("global_count = ", global_count); 

    printf("\n\n\n");

    auto task3 = MyAsyncTask();
    folly::coro::blockingWait(std::move(task3).scheduleOn(&get_executor_2()));
    debug("a = ", global_value["a"], " b = ", global_value["b"]);
    debug("global_count = ", global_count); 

    printf("\n\n\n");

    folly::coro::blockingWait(getLock3().scheduleOn(&get_executor_2()));

    printf("\n\n\n");
    
    folly::coro::blockingWait(getLock2().scheduleOn(&get_executor_2())); // 这里会死锁

    return 0;
}