/*
 * 1. 左值和右值
 *    左值可以取地址，位于等号左边；右值没法取地址，位于等号右边。
 * 2. 左值引用和右值引用
 *    引用是在传参的时候避免拷贝的
 *    左值引用：能指向左值不能指向右值的引用（左值引用不能指向右值，例如常量5其实就是右值，int &ref_a = 5; 会报错）
 *    特例(const左值引用): const int& ref_a = 5; 
 *    右值引用：
 * 
*/ 

#include <iostream>
#include <cstdarg>
#include "rightvalue.h"

namespace RAII {
    void moreChange(int&& right_value) {
        right_value += 2;
        std::cout << "moreChange: " << right_value << std::endl;
    }

    void change(int&& right_value) {
        right_value += 1;
        std::cout << "change: " << right_value << std::endl;

        moreChange(std::forward<int>(right_value));
    }

    // leftChange是可以接收常量的右值的, 这样才能让push_back(5)可以执行
    // 缺点: 传入的只能是const int的引用, 无法让对象自我修改, 缺少了灵活性
    void leftChange(const int& value) {
        std::cout << "leftChange: " << value << std::endl;
    }

    void rightValueFunc() {
        int a = 5;
        int& ref_a = a;
        // 这里本质上就是C++的拷贝构造函数和赋值构造函数的可能性 
        // 目的是支持函数 void push_back(const int& val);
        // 不允许这种情况会导致上面的函数直接报错
        const int& ref_b = 5;

        int&& ref_c_right = 5;
        ref_c_right = 6; // 直接修改右值


        // std::move函数: 进行左右值的转换
        int&& ref_a_right = std::move(ref_a); // 这个函数类似于 static_cast<T&&>(lvalue)
        
        // change(a);  // 编译失败, 本身就是一个左值
        // change(ref_a); // 编译失败, 左值引用本身就是一个左值
        // change(ref_a_right); // 编译失败, 右值引用本身就是一个左值

        // 以下都可以通过编译, 都是对5对应的空间进行操作
        change(std::move(a));
        std::cout << "&a: " <<  &a << "; a: " << a << std::endl;
        change(std::move(ref_a));
        std::cout << "&ref_a: " << &ref_a << "; ref_a: " << ref_a << std::endl;
        change(std::move(ref_a_right));
        std::cout << "&ref_a_right: " << &ref_a_right << "; ref_a_right: " << ref_a_right << std::endl;
        
        // 这里是新开了一个内存空间用于存放5, 
        change(5);
        std::cout << "&a: " <<  &a << "; a: " << a << std::endl;
        
    }
    
    Array::Array(int _s) : size(_s) {
        this->data = new int[_s];
    }

    // 为什么必须用const Array&: 如果只是用Array &, 那么外部传参必须传入左值, 比如在外面去获得搞一个引用才行
    // Array a(2) 这样的操作就无法直接实现, 因为常量2是一个右值
    Array::Array(const Array& other) { 
        this->size = other.size;
        delete[] this->data;
        this->data = new int[this->size];
        for (int i = 0; i < size; i++) {
            this->data[i] = other.data[i];
        }
    }

    Array& Array::operator=(const Array& other) {
        if (this != &other) { // 先传入的引用量取地址, 和this指针进行比较, 如果地址相同就不管了
            this->size = other.size;
            delete[] this->data;
            this->data = new int[this->size];
            for (int i = 0; i < size; i++) {
                this->data[i] = other.data[i];
            }
        }
        // 返回当前地址的一个解引用, 如果返回值是Array就是返回一个拷贝, 如果返回值是Array&就是返回自身的一个引用
        return *this;
    }

    // 为什么必须用Array&&: 
    // 1. 如果只是用Array &, 那么外部传参必须传入左值, 比如在外面去获得搞一个引用才行
    //    Array a(2) 这样的操作就无法直接实现, 因为常量2是一个右值
    // 2. 如果只是用const Array &, 就无法设置other.data = nullptr, 无法避免other对象析构的时候delete[]实际的内存!
    Array::Array(Array&& other) {
        this->size = other.size;
        this->data = other.data;
        // delete[] other.data // 这个操作是非常错误的, delete[]是释放掉对象的内存, 但是移动构造函数就是要让原来的对象的内存移动过来给我使用的
        other.data = nullptr; // 设置为nullptr就是为了在other对象析构的时候, 不去delete[]实际的内存!
    }

    Array::~Array() {
        size = 0;
        delete[] data;
    }

    void Array::setData(const int& count, ...) {
        va_list args;
        va_start(args, count);
        
        for (int i = 0; i < count && i < this->size; i++) {
            this->data[i] = va_arg(args, int);
        }
        va_end(args);
    }

    void Array::print() const {
        std::cout << "print: ";
        for (int i = 0; i < this->size; i++) {
            std::cout << this->data[i] << " ";
        }
        std::cout << std::endl;
    }
};