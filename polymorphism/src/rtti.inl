#include <iostream>
using namespace std;



namespace Polymorphism {
    int global_var = 200;
    
    inline void Base::print() {
        cout << "Base::print" << endl;
    }

    inline void Deride::print() {
        cout << "Deride::print" << endl;
    }

    inline void RunRTTI() {
        Base* b = new Deride();
        cout << typeid(b).name() << endl; // 获取的是指针的类型, 此时指针是Base* 声明的
        // 获取的是指针对应的实例的类型
        // 1. 当类中不存在虚函数的时候，typeid()整个流程都是编译阶段执行的，是静态类型，输出是Base的类型
        // 2. 当类中存在虚函数，typeid()是运行时才能确定类型，是动态类型，输出是Deride的类型
        cout << typeid(*b).name() << endl; 

        { // static_cast
        // 任何编写程序时能够明确的类型转换都可以使用static_cast
        // 不提供运行时的检查，所以叫static_cast，因此，需要在编写程序时确认转换的安全性。
        // 1. 常用在基本类型的转换 int = char = enum
        // 2. 常用在子类对象的指针/引用转化为父类的指针/引用（安全转换）
        // 3. 父类对象的指针/引用转化为子类的指针/引用（不安全转换，例子: void*转化为int*）
        // 不管如何, 编译器都不会报warning和error
            int i = 1, j = 2;
            double slope = static_cast<double>(j) / i;
            cout << "slope: " << slope << endl;

            
            void* p = &slope;
            double* slope_ptr = static_cast<double*>(p);
            cout << "*slope_ptr: " << *slope_ptr << endl;
        }

        { // dynamic_cast
        // 提供运行时的检查，会额外消耗一些性能，但是具有更高的安全性，主要的安全检查在父类对象的指针/引用转化为子类的指针/引用
        // 特殊的地方：
        // 1. 只能针对指针和引用进行dynamic_cast
            Deride* slope = dynamic_cast<Deride*>(b);
            slope->print();
            // cout << "Deride *slope: " << << endl;
            cout << "typeid slope: " << typeid(slope).name() << endl;
            cout << "typeid *slope: " << typeid(*slope).name() << endl;
            
        }

        { // 
        // 特殊的地方: 
        // 1. 可以将const、volatile和__unaligned的类型也进行转换
        //      volatile：禁止编译器进行优化的对象, 系统总是重新从内存读取数据
        //      __unaligned: 
            const char *pc = "12bsdadasd";
            char* p = const_cast<char*>(pc);
            cout << "p: " << p << endl;
        }
        
        { // reinterpret_cast
        // 解决dynamic_cast不能转化基础类型的问题
        // 非常激进的指针类型转换，在编译期完成，可以转换任何类型的指针，所以极不安全。非极端情况不要使用。
            int smallendnum = 0x00636261;
            // reinterpret_cast内容
            int *smallendptr = &smallendnum;
            // reinterpret_cast可以将一个对象转化为char*字符流
            char* ptr = reinterpret_cast<char*>(smallendptr);
            
            cout << "*smallendptr: " << *smallendptr << endl;
            // cout << "*ptr: " << hex << *ptr << endl;
            cout << "ptr: " << ptr << endl; // 直接使用ptr就可以获取其指向的内容, 而且会转化为char
            // cout << "static_cast<void*> ptr: " << static_cast<void*>(ptr) << endl;
        }
    }

};
