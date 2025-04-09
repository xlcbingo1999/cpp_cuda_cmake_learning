#ifndef COROUTINEUSE_DISPATCHER_EXECUTOR_H_
#define COROUTINEUSE_DISPATCHER_EXECUTOR_H_

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

#endif //COROUTINEUSE_DISPATCHER_EXECUTOR_H_