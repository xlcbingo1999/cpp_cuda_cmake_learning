#include "file.h"
#include "mutex.h"
#include "shared_ptr_used.h"
#include "weak_ptr_used.h"
#include "shared_ptr_outside.h"
#include "simple_shared_ptr.h"
#include "pod.h"
#include "deepcopy.h"
#include "rightvalue.h"
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;
int shared_data = 0;



void increment() {
    for (int i = 0; i < 10000; i++) {
        RAII::LockGuard lock(mtx);
        ++shared_data;

        // 这个作用域结束之后就会调用析构函数把mtx解锁了
    }
}

int main() {
    RAII::File myFile("./resource/a.txt");
    if (myFile.getHandle().is_open()) {
        std::cout << "File is open" << std::endl;
    } else {
        std::cout << "Failed to open" << std::endl;
    }

    std::thread t1(increment);
    std::thread t2(increment);
    
    t1.join();
    t2.join();
    
    std::cout << "Shared data: " << shared_data << std::endl;

    { // 多写作用域区间对提升代码有帮助
        // make_shared<T>() 是工厂模式, 根据T模板确定返回的类型
        std::shared_ptr<RAII::SharedPtrUsed> sharedptrused = std::make_shared<RAII::SharedPtrUsed>();
        {
            std::shared_ptr<RAII::SharedPtrUsed> ptr2 = sharedptrused; // 利用赋值构造函数去创建了新的智能指针, 指向同一个区域, 不会去拷贝一个新的空间
            sharedptrused->doSth();
            ptr2->doSth();

            std::cout << "use_count: " << sharedptrused.use_count() << std::endl;
        }
        std::cout << "use_count: " << sharedptrused.use_count() << std::endl;

        
        // 指针的operator bool()重载方法, 用于检查指针是否为空
        if (sharedptrused) {
            std::cout << "sharedptrused not null" << std::endl;
        } else {
            std::cout << "sharedptrused is null" << std::endl;
        }

        // 释放指针的所有权, 将指针的内容设置为nullptr
        sharedptrused.reset();

        std::shared_ptr<int> ptrint1 = std::make_shared<int>(42);
        std::shared_ptr<int> ptrint2 = std::make_shared<int>(21);
        // 用于交换两个指针的内容, 在写排序算法的时候应该会很有用
        ptrint1.swap(ptrint2);
        // operator* 重载操作符可以获取指针指向的对象
        std::cout << "*ptrint1: " << *ptrint1 << "; *ptrint2: " << *ptrint2 << std::endl; 
    }

    {
        // 模拟循环引用的场景, 析构函数不会被调用
        std::shared_ptr<RAII::PtrA> a = std::make_shared<RAII::PtrA>();
        std::shared_ptr<RAII::PtrB> b = std::make_shared<RAII::PtrB>();
        a->b_ptr = b;
        b->a_ptr = a;
    }

    {
        int ini = 10;
        auto d{RAII::SomeData::Create(ini)};
        d->NeedCallSomeAPI();
    }

    {
        RAII::SimpleSharedPtr<RAII::MyClass> ptr1(new RAII::MyClass); // 调用自定义构造函数
        {
            RAII::SimpleSharedPtr<RAII::MyClass> ptr2 = ptr1; // 调用赋值构造函数
            std::cout << "ptr1.use_count(): " << ptr1.use_count() << std::endl;
        }
        std::cout << "ptr1.use_count(): " << ptr1.use_count() << std::endl;
    }

    {
        RAII::PodClass* pod = RAII::GetClass(1, 2, 'c');
        std::cout << pod->a << " " << pod->b << " " << pod->addr << std::endl;
    }

    {
        RAII::DeepCopyClass a("beijing");
        a.print();
        RAII::DeepCopyClass b("shanghai");
        b.print();
        a = b;
        a.print();

        b.setstring("hangzhou").setstring("guangzhou");
        b.print();

        RAII::DeepCopyClass c(a);
        c.print();
    }

    { // 右值, 移动构造, 完美转发
        RAII::rightValueFunc();

        RAII::Array* a = new RAII::Array(2);
        a->setData(2, 4, 5);
        a->print();

        RAII::Array b(std::move(*a)); // 右值赋值后, a就变成空了
        delete a; // 手动释放a对应的内存, 此时a变成一个悬空指针
        b.print(); // 此时b接管了a的内存空间
    }
}