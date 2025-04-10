#ifndef COROUTINEUSE_DISPATCHER_FUTUREAWAITER_H_
#define COROUTINEUSE_DISPATCHER_FUTUREAWAITER_H_

#include "normal_awaiter.h"

template <typename R>
struct FutureAwaiter: public Awaiter<R> {
    using ResultType = R; // 把具体的类型映射出来使用

    explicit FutureAwaiter(std::future<R> &&future): _future(std::move(future)) {}
    FutureAwaiter(FutureAwaiter &&rhs): _future(std::move(rhs._future)) {}
    FutureAwaiter(FutureAwaiter &) = delete;
    FutureAwaiter& operator=(FutureAwaiter &) = delete;

protected:
    void after_suspend() override {
        std::thread([this] () {
            this->resume(this->_future.get()); // coroutine挂起后的操作就是开一个线程阻塞获取结果
        }).detach();
    }

private:
    std::future<R> _future;
};

#endif //COROUTINEUSE_DISPATCHER_FUTUREAWAITER_H_