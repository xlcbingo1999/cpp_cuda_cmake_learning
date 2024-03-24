class Base {
public:
    Base();
    ~Base();
};

class Base1 {
public:
    Base1();
    ~Base1();
};

class Base2 {
public:
    Base2();
    ~Base2();
};

class Base3 {
public:
    Base3();
    ~Base3();
};

// C++的多重继承: 
//   派生类的成员包含了基类的成员和该类本身的成员
//   多重继承会出现二义性的问题, 需要用Base1::print()和Base2::print()限制其二义性!
//   构造函数: 先执行基类的构造函数, 然后再执行派生类的构造函数
//   析构函数: 先执行派生类的析构函数，然后再执行基类的析构函数
// 初始化顺序: 优先虚继承
class MyClass: public virtual Base3, public Base1, public virtual Base2 {
public:
    MyClass(int _num1, int _num2);
    ~MyClass();
private:
    int num1;
    int num2;
    Base base;
};

class TempBase {
public:
    TempBase(int _t);
    ~TempBase();
    void print();
// private:
    int temp;
};

class TempBase1: public TempBase {
public:
    TempBase1(int _t1, int _t2);
    ~TempBase1();
    void print();
// private:
    int temp1;
};

class TempBase2: public TempBase {
public:
    TempBase2(int _t1, int _t2);
    ~TempBase2();
    void print();
// private:
    int temp2;
};

class MyTempClass: public TempBase1, public TempBase2 {
public:
    MyTempClass(int _t11, int _t21, int _t2, int _t3, int _t4);
    ~MyTempClass();
    void print(); // 发生在不同类之间的一个隐藏
// private:
    int temp3;
};