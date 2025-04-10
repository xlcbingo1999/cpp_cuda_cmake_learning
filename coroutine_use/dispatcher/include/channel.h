#ifndef COROUTINEUSE_DISPATCHER_CHANNEL_H_
#define COROUTINEUSE_DISPATCHER_CHANNEL_H_

#include "reader_awaiter.h"
#include "writer_awaiter.h"
#include <condition_variable>
#include <exception>
#include <queue>
#include <list>

template <typename ValueType>
class Channel {
public:

    struct ChannelClosedException: std::exception {
        const char *what() const noexcept override {
            return "Channel is close!";
        }
    };

    explicit Channel(long unsigned int capacity = 0): buffer_capacity(capacity) {
        _is_active.store(true, std::memory_order_relaxed);
    }
    Channel(Channel &&) = delete;
    Channel(Channel &) = delete;
    Channel& operator=(Channel &) = delete;
    ~Channel() {
        close();
    }
    
    bool is_active() {
        return _is_active.load(std::memory_order_relaxed);
    }

    void check_close() {
        if (!is_active()) {
            throw ChannelClosedException();
        }
    }

    void close() {
        bool expect = true;
        // 只有_is_active为真的时候才能返回true
        if (_is_active.compare_exchange_strong(expect, false, std::memory_order_relaxed)) {
            clean_up();
        }
    }

    auto write(ValueType value) {
        check_close();
        return WriterAwaiter<ValueType>(this, value); // 本质上就是把awaiter创建出来然后返回，应该是给co_await使用的？
    }

    auto operator<<(ValueType value) {
        return write(value);
    }

    auto read() {
        check_close();
        return ReaderAwaiter<ValueType>(this);
    }

    auto operator>>(ValueType &value_ref) {
        auto awaiter = read();
        awaiter.set_p_value(&value_ref); // 需要将地址写入awaiter
        return awaiter;
    }

    void try_push_reader(ReaderAwaiter<ValueType> *read_awaiter) {
        std::unique_lock lock(channel_lock);
        check_close();

        // 读取操作优先从 buffer 读取
        if (!buffer.empty()) {
            auto value = buffer.front();
            buffer.pop();

            // 因为弹出了一个buffer的内容，所以如果有writer阻塞着，就赶紧写入
            if (!writer_list.empty()) {
                auto writer = writer_list.front();
                writer_list.pop_front();
                buffer.push(writer->get_value());
                
                lock.unlock();
                
                writer->resume(); // 让writer接着恢复执行
            } else {
                lock.unlock();
            }

            read_awaiter->resume(value);
            return ;
        }

        // 没有buffer的话就从writer_list去拿
        if (!writer_list.empty()) {
            auto writer = writer_list.front();
            writer_list.pop_front();
            lock.unlock();
            
            read_awaiter->resume(writer->get_value());
            writer->resume();
            return ;
        }

        // 如果连writer_list都是空的，直接就加入等待队列中
        reader_list.push_back(read_awaiter);
    }

    void try_push_writer(WriterAwaiter<ValueType> *write_awaiter) {
        std::unique_lock lock(channel_lock);
        check_close();

        // 如果写入的时候，读队列是有挂起的，需要找一个去消费
        if (!reader_list.empty()) {
            auto reader = reader_list.front();
            reader_list.pop_front();
            lock.unlock();

            reader->resume(write_awaiter->get_value()); // 直接让reader进行消费
            write_awaiter->resume(); // writer完成操作，也可以恢复执行
            return ;
        }

        // 如果没有读者需要消费，需要根据buffer情况进行处理
        // buffer未满，writer是不需要阻塞的
        if (buffer.size() < buffer_capacity) {
            buffer.push(write_awaiter->get_value());
            lock.unlock();

            write_awaiter->resume(); // writer完成操作，可以恢复执行
            return ;
        }

        writer_list.push_back(write_awaiter); // 放入等待消费的队列中
    }

    void remove_reader(ReaderAwaiter<ValueType> *read_awaiter) {
        std::lock_guard lock(channel_lock);
        reader_list.remove(read_awaiter);
    }

    void remove_writer(WriterAwaiter<ValueType> *write_awaiter) {
        std::lock_guard lock(channel_lock);
        writer_list.remove(write_awaiter);
    }

private:
    long unsigned int buffer_capacity;
    std::queue<ValueType> buffer; // 环状队列会更好
    std::list<WriterAwaiter<ValueType> *> writer_list; // 挂起的写coroutine
    std::list<ReaderAwaiter<ValueType> *> reader_list; // 挂起的读coroutine

    std::atomic<bool> _is_active;
    std::mutex channel_lock;
    std::condition_variable channel_condition;

    void clean_up() {
        std::lock_guard lock(channel_lock);
        for (auto writer: writer_list) {
            writer->resume(); // 关闭的时候需要让writer都恢复一下，把该弄完的都搞定
        }
        writer_list.clear();
        
        for (auto reader: reader_list) {
            reader->resume(); // 关闭的时候需要让reader都恢复一下，把该弄完的都搞定
        }
        reader_list.clear();

        decltype(buffer) empty_buffer;
        std::swap(buffer, empty_buffer);
    }    
};

#endif //COROUTINEUSE_DISPATCHER_CHANNEL_H_