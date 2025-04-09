
#ifndef COROUTINEUSE_TASK_TASKAWAITER_H_
#define COROUTINEUSE_TASK_TASKAWAITER_H_


#include <coroutine>
#include <utility>
template <typename ResultType>
struct Task;

template <typename ResultType>
struct TaskAwaiter {
    explicit TaskAwaiter(Task<ResultType> &&task) noexcept : task(std::move(task)) {}
    TaskAwaiter(TaskAwaiter &&completion) noexcept : task(std::exchange(completion.task, {})) {}
    TaskAwaiter(TaskAwaiter &) = delete;
    TaskAwaiter& operator=(TaskAwaiter &) = delete;

    constexpr bool await_ready() const noexcept { return false; } // 需要挂起

    ResultType await_resume() noexcept {
        return task.get_result();
    }

    void await_suspend(std::coroutine_handle<> handle) noexcept {
        task.finally([handle] () {
            handle.resume();
        });
    }
private:
    Task<ResultType> task;
};

#endif //COROUTINEUSE_TASK_TASKAWAITER_H_