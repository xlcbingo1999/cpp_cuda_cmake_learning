#include <iostream>

namespace RAII {
    class MyClass {
    public:
        MyClass();
        ~MyClass();
        void do_something();
    };

    template <typename T> // typename可以传入任何的类型
    class SimpleSharedPtr {
    public:
        explicit SimpleSharedPtr(T* ptr = nullptr);
        SimpleSharedPtr(const SimpleSharedPtr& other);
        SimpleSharedPtr& operator=(const SimpleSharedPtr& other);
        ~SimpleSharedPtr();

        T& operator*() const; // 获取指针的值, 此时返回值的引用是合理的, 还能对其进行一些赋值相关的操作
        T* operator->() const; // 用于调用指针的方法和函数, 此时返回的是原始指针, 可以进一步进行操作
        T* get() const; // 用于获取原始指针
        size_t use_count() const;
        size_t* use_count_ptr() const;
    private:
        void release();
        T* ptr_;
        size_t* count_; // 为何这里要保存一个指针类型?
    };

};