// 场景: 菱形继承, 普通的初始化方案会让最后的派生类带有最初基类的多份成员变量的拷贝
// 优化: 虚继承, 要求是让最后的派生类来对最初基类的成员变量进行属性赋值, 而不是让多个菱形派生类进行初始化

namespace VirturalMulti {
    class TempBase {
    public:
        // 构造函数不能是虚函数
        // 1. 构造函数还没调用之前，此时class还没初始化好虚函数表
        // 2. 构造函数是会被自动调用的，没必要进行虚函数重写Override
        TempBase(int _t); 
        ~TempBase();
        void print();
    // private:
        int temp;
    };

    class TempBase1: virtual public TempBase {
    public:
        TempBase1(int _t1, int _t2);
        ~TempBase1();
        void print();
    // private:
        int temp1;
    };

    class TempBase2: virtual public TempBase {
    public:
        TempBase2(int _t1, int _t2);
        ~TempBase2();
        void print();
    // private:
        int temp2;
    };

    class MyTempClass: public TempBase1, public TempBase2 {
    public:
        MyTempClass(int _t, int _t1, int _t2, int _t3); // 需要由最后的派生类来进行最终
        ~MyTempClass();
        void print(); // 发生在不同类之间的一个隐藏
    // private:
        int temp3;
    };
};