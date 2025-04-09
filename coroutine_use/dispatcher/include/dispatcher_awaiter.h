#ifndef COROUTINEUSE_DISPATCHER_DISPATCHERAWAITER_H_
#define COROUTINEUSE_DISPATCHER_DISPATCHERAWAITER_H_

#include "executor.h"
#include <coroutine>
#include <utility>

struct DispatcherAwaiter {
    explicit DispatcherAwaiter(AbstractExecutor *executor) noexcept: _executor(executor) {}
    DispatcherAwaiter(DispatcherAwaiter &&rhs) noexcept: _executor(std::exchange(rhs._executor, {})) {}
    DispatcherAwaiter(DispatcherAwaiter &) = delete;
    DispatcherAwaiter& operator=(DispatcherAwaiter &) = delete;

    constexpr bool await_ready() const noexcept {
        return false; // 要suspend
    }

    void await_resume() noexcept {
        
    }

    void await_suspend(std::coroutine_handle<> handle) noexcept {
        _executor->execute([handle] () {
            handle.resume(); // 核心就是挂起的时候就要让executor决定resume()的位置即可
        });
    }
private:
    AbstractExecutor *_executor;
};

#endif //COROUTINEUSE_DISPATCHER_DISPATCHERAWAITER_H_