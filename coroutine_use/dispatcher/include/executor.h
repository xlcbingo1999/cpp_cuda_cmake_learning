#ifndef COROUTINEUSE_DISPATCHER_EXECUTOR_H_
#define COROUTINEUSE_DISPATCHER_EXECUTOR_H_

#include "delayed_executable.h"
#include <functional>
#include <thread>
#include <future>
#include <queue>


class AbstractExecutor {
public:
    virtual void execute(std::function<void()> &&func) = 0; // 虚函数等待实现
};

class NoopExecutor: public AbstractExecutor {
public:
    void execute(std::function<void()> &&func) override {
        func(); // 直接就是阻塞调用
    }
};

class NewThreadExecutor: public AbstractExecutor {
public:
    void execute(std::function<void()> &&func) override {
        std::thread(func).detach(); // 创建thread然后detach去执行
    }
};

class AsyncExecutor: public AbstractExecutor {
public:
    void execute(std::function<void()> &&func) override {
        auto future = std::async(func);
    }
};

class LooperExecutor: public AbstractExecutor {
private:
    std::atomic<bool> is_active; // 主要是标示executor是否存活
    std::queue<std::function<void()>> executable_queue; // FCFS队列
    std::mutex queue_lock; // 队列锁
    std::condition_variable queue_condition; // 用于队列接收到新的func时进行通知使用
    std::thread work_thread; // 单线程处理即可
    

    void run_loop() {
        while (is_active.load(std::memory_order_relaxed) || !executable_queue.empty() ) {
            std::unique_lock lock(queue_lock); // 从队列拿内容需要加锁
            
            if (executable_queue.empty()) { // double-check
                queue_condition.wait(lock); // 把锁交付给queue_condition，直到收到通知

                // 收到的通知可能是关闭queue_condition的通知，因此还要再check一下
                if (executable_queue.empty()) {
                    continue; // 再走到while判断即可
                }
            }

            auto func = executable_queue.front();
            executable_queue.pop();
            lock.unlock();

            func(); // 执行
        }
    }
public:
    LooperExecutor() {
        is_active.store(true, std::memory_order_relaxed);
        work_thread = std::thread(&LooperExecutor::run_loop, this); // 需要传入this指针
    }

    ~LooperExecutor() {
        shutdown(false);
        join();
    }

    void shutdown(bool wait_for_complete = true) {
        is_active.store(false, std::memory_order_relaxed);    
        if (!wait_for_complete) {
            // 直接清空整个任务队列就行
            std::unique_lock lock(queue_lock);
            decltype(executable_queue) empty_queue;
            std::swap(executable_queue, empty_queue);
            lock.unlock();
        }
        queue_condition.notify_all(); // 通知worker_thread说此时executor结束了，会导致run_loop结束
    }

    void join() {
        if (work_thread.joinable()) {
            work_thread.join();
        }
    }

    void execute(std::function<void()> &&func) override {
        std::unique_lock lock(queue_lock);
        if (is_active.load(std::memory_order_relaxed)) {
            executable_queue.push(func);
            lock.unlock();

            queue_condition.notify_one(); // 不需要加锁
        }
    }
};

class SharedLooperExecutor: public LooperExecutor { // 只有使用这个executor，才能让所有的coroutine都使用同一个worker_thread
public:
    void execute(std::function<void()> &&func) override {
        static LooperExecutor shared_looper_executor; // 静态变量，使得全局只有一个
        shared_looper_executor.execute(std::move(func)); // 移动一下变成右值
    }
};

class SleepExecutor {
public:
    SleepExecutor() {
        is_active.store(true, std::memory_order_relaxed);
        work_thread = std::thread(&SleepExecutor::run_loop, this);
    }

    ~SleepExecutor() {
        shutdown(false);
        join();
    }

    void execute(std::function<void()> &&func, long long delay) {
        delay = delay < 0 ? 0 : delay;
        std::unique_lock lock(queue_lock);
    
        if (is_active.load(std::memory_order_relaxed)) {
            bool need_notify = executable_queue.empty() || executable_queue.top().get_delay_time() > delay;
            executable_queue.push(DelayedExecutable(std::move(func), delay));
            lock.unlock();
            if (need_notify) {
                queue_condition.notify_one();
            }
        }
    }
    
    void shutdown(bool need_wait_finished = true) {
        is_active.store(false, std::memory_order_relaxed);
        if (!need_wait_finished) {
            // 直接清空队列
            std::unique_lock lock(queue_lock);
            decltype(executable_queue) empty_queue;
            std::swap(executable_queue, empty_queue);
            lock.unlock();
        }
        queue_condition.notify_all();
    }

    void join() {
        if (work_thread.joinable()) {
            work_thread.join();
        }
    }
private:
    std::priority_queue<DelayedExecutable, std::vector<DelayedExecutable>, DelayedExecutableCompare> executable_queue;
    std::mutex queue_lock;
    std::condition_variable queue_condition;
    
    std::atomic<bool> is_active;
    std::thread work_thread;

    void run_loop() {
        while (is_active.load(std::memory_order_relaxed) || !executable_queue.empty()) {
            // 获取锁
            std::unique_lock lock(queue_lock);
            // double-check
            if (executable_queue.empty()) {
                // 等待任务到达，等待通知的操作，因此当队列从空到非空，需要notify
                queue_condition.wait(lock);

                // 如果收到的事件是关闭事件，需要额外处理
                if (executable_queue.empty()) {
                    continue;
                }
            }

            auto executable = executable_queue.top();
            long long delay = executable.get_delay_time();
            if (delay > 0) { // 此时还不可以执行，直接wait delay的事件，如果在delay前收到别的事件，则需要额外处理
                // 这是一个非常优雅的操作，只需要阻塞一个thread即可实现sleep操作！
                auto status = queue_condition.wait_for(lock, std::chrono::milliseconds(delay));

                if (status != std::cv_status::timeout) {
                    // 如果收到的事件不是timeout，说明队列可能进入了优先级更高的任务，需要优先处理，因此这里continue出去即可
                    continue;
                }
            }
            // 如果没有delay，或者已经wait delay的时间后
            executable_queue.pop(); // 出队
            lock.unlock();
            executable(); // 执行任务
        }
    }
};

#endif //COROUTINEUSE_DISPATCHER_EXECUTOR_H_