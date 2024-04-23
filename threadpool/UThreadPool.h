#ifndef CGRAPH_UTHREADOBJECT_H
#define CGRAPH_UTHREADOBJECT_H

#include "./CObject.h"
#include "./UTask.h"
#include <vector>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>

using TaskFunc = std::function<void()>; // 表示函数签名是 void func() 的函数
#define MAX_THREAD_NUM 16
#define INIT_THREAD_NUM 4

class UThreadPool : public CObject {
public:
    UThreadPool(const UThreadPool&) = delete;
    const UThreadPool& operator=(const UThreadPool&) = delete; 


    explicit UThreadPool(int maxThreadNum = MAX_THREAD_NUM) { // 禁止except
        max_thread_num_ = maxThreadNum;
        addThread(max_thread_num_);   
    }

    virtual ~UThreadPool() {
        run_ = false;
        task_cond_.notify_all(); // 让所有的阻塞中的线程都能执行, 然后看到了已经被关闭, 结束线程的run()
        for (std::thread& thd: pool_) {
            if (thd.joinable()) {
                thd.join();
            }
        }
    }

    std::future<int> commit(const UTask& task) {
        auto curTask = std::make_shared<std::packaged_task<int()>>(std::bind(&UTask::run, task));
        std::future<int> future = curTask->get_future();
        {
            std::lock_guard<std::mutex> lock{mtx_}; // lock_guard禁止拷贝构造和移动构造, 也是RAII表示
            task_que_.push([curTask] () {
                (*curTask)(); // 任务队列中的函数就是执行自身的闭包
            });

            if (idle_thread_num_ < 1 && pool_.size() < max_thread_num_) {
                addThread(1); // 有需要就新增
            }
        }

        // 提交一个任务后需要让一个线程来执行
        task_cond_.notify_one();
        return future;
    }

    int run() {
        while (run_) { // 线程池必须启动着
            TaskFunc curFunc = nullptr;
            {
                std::unique_lock<std::mutex> lock{mtx_}; // RAII的锁的包装器, 用于管理获取的锁的生命周期
                task_cond_.wait(lock, [this] {
                    // 当线程池被启动或者任务队列为空的时候, wait()的闭包返回false, 此时会阻塞在这里等待内容
                    // 阻塞的时候, 会自动释放锁, 并将线程交付给其他线程使用
                    return (!run_ || !task_que_.empty());
                });

                if (!run_ && task_que_.empty()) {
                    return 0; // 线程正常结束
                }

                curFunc = std::move(task_que_.front()); // 移动操作, 被移动出来的内容就不再需要了
                task_que_.pop();
            }

            idle_thread_num_--; // 启动一个线程处理
            if (curFunc) {
                curFunc();
            }
            idle_thread_num_++; // 恢复一个线程
        }
        return 0; // 线程正常结束
    }

protected:
    void addThread(int size) {
        for (; pool_.size() < max_thread_num_ && size > 0; size--) {
            pool_.emplace_back(std::thread(&UThreadPool::run, this));
            idle_thread_num_++; // 原子操作
        }
    }
private:
    std::vector<std::thread> pool_;
    std::queue<TaskFunc> task_que_; 
    std::mutex mtx_;
    std::condition_variable task_cond_;
    std::atomic<bool> run_ {true};  // 表示线程池是否正在启动中, 关闭启动需要是原子操作
    std::atomic<int> idle_thread_num_ {0}; // 空闲线程数量
    std::atomic<int> max_thread_num_ {MAX_THREAD_NUM}; // 最多的线程数量
    
};

#endif //CGRAPH_UTHREADOBJECT_H