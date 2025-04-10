#ifndef COROUTINEUSE_DISPATCHER_NORMALAWAITER_H_
#define COROUTINEUSE_DISPATCHER_NORMALAWAITER_H_

#include "executor.h"
#include "result.h"
#include <coroutine>

template <typename R>
struct Awaiter {
    
    void install_executor(AbstractExecutor *executor) {
        _executor = executor;
    }

    constexpr bool await_ready() const noexcept {
        return false; // 要suspend
    }

    R await_resume() {
        before_resume();
        return _result->get_or_throw();
    }

    void await_suspend(std::coroutine_handle<> handle) {
        _handle = handle;
        after_suspend();
    }

    void resume(R value) {
        // coroutine的恢复一定要用调度器去执行
        dispatch([this, value] () {
            _result = Result<R>(static_cast<R>(value));
            _handle.resume();
        });
    }

    void resume_unsafe() {
        // 这个操作没有传入值，此时要求子类在before_resume的时候就要写入 _result或者抛出异常，在channel关闭的时候会使用
        dispatch([this] () {
            _handle.resume();
        });
    }

    void resume_exception(std::exception_ptr &&e) {
        dispatch([this, e] () {
            _result = Result<R>(static_cast<std::exception_ptr>(e));
            _handle.resume();
        });
    }

protected: // 只有子类可见的内容
    std::optional<Result<R>> _result{};

    virtual void after_suspend() {}

    virtual void before_resume() {}

private:
    AbstractExecutor *_executor = nullptr;
    std::coroutine_handle<> _handle = nullptr;

    void dispatch(std::function<void()> &&f) {
        // 通用的执行函数
        if (_executor) {
            _executor->execute(std::move(f));
        } else {
            f();
        }
    }
};


template <>
struct Awaiter<void> {
    
    void install_executor(AbstractExecutor *executor) {
        _executor = executor;
    }

    constexpr bool await_ready() const noexcept {
        return false; // 要suspend
    }

    void await_resume() {
        before_resume();
        _result->get_or_throw();
    }

    void await_suspend(std::coroutine_handle<> handle) {
        _handle = handle;
        after_suspend();
    }

    void resume() {
        // coroutine的恢复一定要用调度器去执行
        dispatch([this] () {
            _result = Result<void>();
            _handle.resume();
        });
    }

    void resume_unsafe() {
        // 这个操作没有传入值，此时要求子类在before_resume的时候就要写入 _result或者抛出异常，在channel关闭的时候会使用
        dispatch([this] () {
            _handle.resume();
        });
    }

    void resume_exception(std::exception_ptr &&e) {
        dispatch([this, e] () {
            _result = Result<void>(static_cast<std::exception_ptr>(e));
            _handle.resume();
        });
    }

protected: // 只有子类可见的内容
    std::optional<Result<void>> _result{};

    virtual void after_suspend() {}

    virtual void before_resume() {}

private:
    AbstractExecutor *_executor = nullptr;
    std::coroutine_handle<> _handle = nullptr;

    void dispatch(std::function<void()> &&f) {
        // 通用的执行函数
        if (_executor) {
            _executor->execute(std::move(f));
        } else {
            f();
        }
    }
};

#endif //COROUTINEUSE_DISPATCHER_NORMALAWAITER_H_