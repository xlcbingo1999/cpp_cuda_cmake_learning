#ifndef COROUTINEUSE_DISPATCHER_READERAWAITER_H_
#define COROUTINEUSE_DISPATCHER_READERAWAITER_H_

#include "executor.h"
#include <coroutine>

template<typename ValueType>
class Channel;

template <typename ValueType>
struct ReaderAwaiter {
    ReaderAwaiter(Channel<ValueType> *channel): _channel(channel) {}
    ReaderAwaiter(ReaderAwaiter &&other) noexcept {
        // 这里要实现移动构造函数，应该是在做一些内存转移的操作时候需要被调用
        this->_channel = std::exchange(other._channel, nullptr);
        this->_executor = std::exchange(other._executor, nullptr);
        this->_value = other._value;
        this->_p_value = std::exchange(other._p_value, nullptr);
        this->_handle = other._handle;
    }

    ~ReaderAwaiter() {
        if (_channel) { // 如果_channel 存在但是析构掉reader coroutine，说明此时reader还没被恢复但要被销毁，因此需要主动做一次unregister
            _channel->remove_reader(this); 
        }
    }

    constexpr bool await_ready() const noexcept {
        return false; // 要suspend
    }

    ValueType await_resume() {
        // 恢复执行的时候需要返回结果，因为是读
        _channel->check_close();
        _channel = nullptr; // reader被恢复执行后，_channel也不需要维护着了!
        return _value;
    }

    auto await_suspend(std::coroutine_handle<> handle) {
        // 挂起的时候需要将这个操作传递给channel实例，等待恢复
        this->_handle = handle;
        _channel->try_push_reader(this);
    }

    void resume(ValueType value) {
        // [正常读] 恢复该coroutine时需要调用resume函数
        this->_value = value;
        if (_p_value) {
            *_p_value = value;
        }
        resume();
    }

    void resume() {
        // [关闭channel] 恢复该coroutine时需要调用resume函数
        if (_executor) {
            _executor->execute([this] () {
                _handle.resume();
            });
        } else {
            _handle.resume();
        }
    }

    void set_p_value(ValueType *rhs) {
        _p_value = rhs;
    }

    void set_executor(AbstractExecutor *executor) {
        _executor = executor;
    }
private:
    Channel<ValueType> *_channel;
    AbstractExecutor *_executor = nullptr; // 可以不传入调度器，这样就是在当前thread进行
    ValueType _value;
    ValueType* _p_value = nullptr; // 将变量的地址传进来，coroutine恢复时候写入变量对应的内存
    std::coroutine_handle<> _handle;
};

#endif //COROUTINEUSE_DISPATCHER_READERAWAITER_H_