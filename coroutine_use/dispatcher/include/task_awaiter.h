
#ifndef COROUTINEUSE_DISPATCHER_TASKAWAITER_H_
#define COROUTINEUSE_DISPATCHER_TASKAWAITER_H_

#include "executor.h"
#include <coroutine>
#include <utility>

template <typename ResultType, typename Executor>
struct Task;

template <typename ResultType, typename Executor>
struct TaskAwaiter {
    explicit TaskAwaiter(Task<ResultType, Executor> &&task, AbstractExecutor *executor) noexcept : task(std::move(task)), _executor(executor) {}
    TaskAwaiter(TaskAwaiter &&completion) noexcept : task(std::exchange(completion.task, {})) {}
    TaskAwaiter(TaskAwaiter &) = delete;
    TaskAwaiter& operator=(TaskAwaiter &) = delete;

    constexpr bool await_ready() const noexcept { return false; } // 需要挂起

    ResultType await_resume() noexcept {
        return task.get_result();
    }

    void await_suspend(std::coroutine_handle<> handle) noexcept {
        task.finally([handle, this] () {
            this->_executor->execute([handle] () {
                handle.resume();
            });
        });
    }
private:
    Task<ResultType, Executor> task;
    AbstractExecutor *_executor;
};

#endif //COROUTINEUSE_DISPATCHER_TASKAWAITER_H_