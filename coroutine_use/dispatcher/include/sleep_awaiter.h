
#ifndef COROUTINEUSE_DISPATCHER_SLEEPAWAITER_H_
#define COROUTINEUSE_DISPATCHER_SLEEPAWAITER_H_

#include "executor.h"
#include <coroutine>
struct SleepAwaiter {
    explicit SleepAwaiter(AbstractExecutor *executor, long long duration) noexcept : _executor(executor), _duration(duration) {}

    bool await_ready() const {
        return false;
    }

    void await_resume() {

    }

    void await_suspend(std::coroutine_handle<> handle) const {
        static SleepExecutor sleep_executor; // 全局只需要维护一个即可

        sleep_executor.execute([this, handle]() {
            this->_executor->execute([handle]() {
                handle.resume();
            });
        }, _duration);
    }
private:
    AbstractExecutor *_executor;
    long long _duration; // sleep时间
};

#endif //COROUTINEUSE_DISPATCHER_SLEEPAWAITER_H_