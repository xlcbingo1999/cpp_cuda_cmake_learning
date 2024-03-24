#include "mutex.h"
#include <mutex>


namespace RAII {
    LockGuard::LockGuard(std::mutex &mtx) : mutex_(mtx) {
        // std::cout << "lock" << std::endl;
        this->mutex_.lock(); // 构造的时候马上就加锁了
    }

    LockGuard::~LockGuard() {
        // std::cout << "unlock" << std::endl;
        this->mutex_.unlock(); // 只有在析构的时候才解锁
    }
};