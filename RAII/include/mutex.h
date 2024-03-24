#include <mutex>

namespace RAII {
    class LockGuard {
    public:
        ~LockGuard();
        // 构造函数的声明带有explicit, 可以避免隐式转换, 函数体中只能显示转换
        // 主要的作用就是避免调用构造函数的参数进行隐式转换
        explicit LockGuard(std::mutex &mtx);
        // 禁止拷贝构造函数, 不能让别人来处理 
        LockGuard(const LockGuard& mtx) = delete;
        // 禁止赋值构造函数
        LockGuard& operator=(const LockGuard& mtx) = delete;
    private:
        std::mutex &mutex_;
    };

};