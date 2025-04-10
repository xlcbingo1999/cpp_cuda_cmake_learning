
#ifndef COROUTINEUSE_DISPATCHER_DELAYEDEXECUTABLE_H_
#define COROUTINEUSE_DISPATCHER_DELAYEDEXECUTABLE_H_

#include <chrono>
#include <functional>

// 这是对 std::function<void()> 的封装，增加了延时执行的功能
class DelayedExecutable {
public:
    DelayedExecutable(std::function<void()> &&func, long long delay): func(std::move(func)) {
        auto now = std::chrono::system_clock::now();
        
        // 转化成ms时间戳
        auto current = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

        // 转化为计划执行的时间
        promise_scheduled_time = current + delay;
    }

    long long get_delay_time() const {
        auto now = std::chrono::system_clock::now();
        
        // 转化成ms时间戳
        auto current = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

        return promise_scheduled_time - current; // 计算还剩多久要执行
    }

    long long get_promise_scheduled_time() const {
        return promise_scheduled_time;
    }

    void operator()() {
        func(); // 执行函数
    }
private:
    std::function<void()> func;
    long long promise_scheduled_time;
};

class DelayedExecutableCompare {
    // 由于需要放入到优先级队列，因此需要写一个比较类
public:
    bool operator()(DelayedExecutable &lhs, DelayedExecutable &rhs) {
        // 越小的越靠前
        return lhs.get_promise_scheduled_time() > rhs.get_promise_scheduled_time();
    }
};


#endif //COROUTINEUSE_DISPATCHER_DELAYEDEXECUTABLE_H_