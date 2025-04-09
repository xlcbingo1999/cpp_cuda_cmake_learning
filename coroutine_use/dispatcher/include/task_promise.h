#ifndef COROUTINEUSE_DISPATCHER_TASKPROMISE_H_
#define COROUTINEUSE_DISPATCHER_TASKPROMISE_H_

#include "result.h"
#include "task_awaiter.h"
#include "dispatcher_awaiter.h"
#include <condition_variable>
#include <exception>
#include <optional>
#include <mutex>
#include <functional>
#include <list>
#include <coroutine>

template <typename ResultType, typename Executor>
struct Task;

template <typename ResultType, typename Executor>
struct TaskPromise {
    // 这里是为了让coroutine第一次挂起的时候就要丢到对应的executor的线程上执行
    // 因此需要 自定义一个 DispatchAwaiter，要求是Task协程启动的时候就要挂起，然后将 consume() 函数的调用也让executor进行管理
    DispatcherAwaiter initial_suspend() noexcept {
        return DispatcherAwaiter{&executor}; // 标准构造即可
    }

    std::suspend_always final_suspend() noexcept {
        return {};
    }
    
    Task<ResultType, Executor> get_return_object() {
        return Task{std::coroutine_handle<TaskPromise>::from_promise(*this)};
    }

    // TaskAwaiter<ResultType> await_transform(Task<ResultType> &&task) {
    //     return TaskAwaiter<ResultType>(std::move(task));
    // }

    // TODO(xlc): 原文这里写了一个 _ResultType 不清楚区别是什么，上面注释的代码也能跑
    template<typename _ResultType, typename _Executor>
    TaskAwaiter<_ResultType, _Executor> await_transform(Task<_ResultType, _Executor> &&task) {
        return TaskAwaiter<_ResultType, _Executor>(std::move(task), &executor);
    }

    void unhandled_exception() {
        std::lock_guard lock(completion_lock);
        result = Result<ResultType>(std::current_exception());
        completion.notify_all(); // 条件变量让所有的thread都知道
        notify_callbacks(); // 执行所有的callback
    }

    void return_value(ResultType value) { // coroutine co_return的时候会返回，此时向cv发送了notify
        std::lock_guard lock(completion_lock);
        result = Result<ResultType>(std::move(value));
        completion.notify_all(); // 条件变量让所有的thread都知道
        notify_callbacks(); // 执行所有的callback
    }

    ResultType get_result() {
        std::unique_lock lock(completion_lock);
        if (!result.has_value()) {
            completion.wait(lock); // 阻塞等待结果
        }
        return result->get_or_throw();
    }

    void on_completed(std::function<void(Result<ResultType>)> &&func) {
        std::unique_lock lock(completion_lock);
        if (result.has_value()) { // 注册的时候发现刚好执行完了，直接回调就行
            auto value = result.value();
            lock.unlock();
            func(value); // 执行回调即可
        } else {
            completion_callbacks.push_back(func); // 正常注册逻辑：异步操作，放入一个缓冲区中，等任务自动完成后就取出来执行
        }
    }
private:
    Executor executor;
    std::optional<Result<ResultType>> result;
    
    std::mutex completion_lock;
    std::condition_variable completion;

    std::list<std::function<void(Result<ResultType>)>> completion_callbacks; // 异步时候需要一个list存完成的内容，再主动推送出去
    void notify_callbacks() {
        auto value = result.value();
        for (auto &callback: completion_callbacks) {
            callback(value);
        }
        completion_callbacks.clear();
    }
};

#endif //COROUTINEUSE_DISPATCHER_TASKPROMISE_H_