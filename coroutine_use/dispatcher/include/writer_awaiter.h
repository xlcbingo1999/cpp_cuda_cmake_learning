#ifndef COROUTINEUSE_DISPATCHER_WRITERAWAITER_H_
#define COROUTINEUSE_DISPATCHER_WRITERAWAITER_H_

#include "executor.h"
#include <coroutine>

template<typename ValueType>
class Channel;

template <typename ValueType>
struct WriterAwaiter {
    WriterAwaiter(Channel<ValueType> *channel, ValueType value): _channel(channel), _value(value) {}
    WriterAwaiter(WriterAwaiter &&other) noexcept {
        // 这里要实现移动构造函数，应该是在做一些内存转移的操作时候需要被调用
        this->_channel = std::exchange(other._channel, nullptr);
        this->_executor = std::exchange(other._executor, nullptr);
        this->_value = other._value;
        this->_handle = other._handle;
    }

    ~WriterAwaiter() {
        if (_channel) { // 如果_channel 存在但是析构掉reader coroutine，说明此时reader还没被恢复但要被销毁，因此需要主动做一次unregister
            _channel->remove_writer(this); 
        }
    }

    constexpr bool await_ready() const noexcept {
        return false; // 要suspend
    }

    void await_resume() noexcept {
        // 恢复执行的时候需要check一下是不是channel的关闭操作
        _channel->check_close();
        _channel = nullptr;
    }

    auto await_suspend(std::coroutine_handle<> handle) noexcept {
        // 挂起的时候需要将这个操作传递给channel实例，等待恢复
        this->_handle = handle;
        _channel->try_push_writer(this); // 把自己传给channel
    }

    void resume() {
        // chanel恢复该coroutine时需要调用resume函数
        if (_executor) {
            _executor->execute([this] () {
                _handle.resume();
            });
        } else {
            _handle.resume(); // 本质上就是把存储起来的handle恢复掉
        }
    }

    ValueType get_value() {
        return _value;
    }

    void set_executor(AbstractExecutor *executor) {
        _executor = executor;
    }
private:
    Channel<ValueType> *_channel;
    AbstractExecutor *_executor = nullptr; // 可以不传入调度器，这样就是在当前thread进行
    ValueType _value;
    std::coroutine_handle<> _handle;

};

#endif //COROUTINEUSE_DISPATCHER_WRITERAWAITER_H_