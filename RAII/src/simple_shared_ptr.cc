#include "simple_shared_ptr.h"

namespace RAII {
    MyClass::MyClass() { 
        std::cout << "MyClass 构造函数\n";
    }
    MyClass::~MyClass() { 
        std::cout << "MyClass 析构函数\n"; 
    }
    void MyClass::do_something() { 
        std::cout << "MyClass::do_something() 被调用\n"; 
    }


    template <typename T>
    SimpleSharedPtr<T>::SimpleSharedPtr(T* ptr) {
        this->ptr_ = ptr;
        this->count_ = ptr ? new size_t(1) : nullptr;
    }

    template <typename T>
    SimpleSharedPtr<T>::SimpleSharedPtr(const SimpleSharedPtr<T>& other) {
        this->ptr_ = other.get(); // 隐藏空间是否可以获取
        this->count_ = other.use_count_ptr();
        if (this->count_) {
            ++(*count_); // 把指针对应的值增加了
        }
    }

    template <typename T>
    SimpleSharedPtr<T>& SimpleSharedPtr<T>::operator=(const SimpleSharedPtr<T>& other) {
        if (this != &other) {
            release(); // 需要先释放指针, 然后再进行处理
            this->ptr_ = other.get();
            this->count_ = other.use_count_ptr();
            if (this->count_) {
                ++(*count_);
            }
        }
        return *this;
    }

    template <typename T>
    SimpleSharedPtr<T>::~SimpleSharedPtr() {
        release();
    }

    template <typename T>
    T& SimpleSharedPtr<T>::operator*() const {
        return *ptr_;
    }

    template <typename T>
    T* SimpleSharedPtr<T>::operator->() const {
        return ptr_;
    }

    template <typename T>
    T* SimpleSharedPtr<T>::get() const {
        return ptr_;
    }

    template <typename T>
    size_t SimpleSharedPtr<T>::use_count() const {
        return this->count_ ? *(this->count_) : 0;
    }

    template <typename T>
    size_t* SimpleSharedPtr<T>::use_count_ptr() const {
        return this->count_;
    }

    template <typename T>
    void SimpleSharedPtr<T>::release() {
        if (count_ && --(*count_) == 0) {
            delete ptr_;
            delete count_;
        }
    }

    // 必须在定义的时候就先声明特化类型, 这样才不会报找不到符号的错误
    template class SimpleSharedPtr<MyClass>;
};