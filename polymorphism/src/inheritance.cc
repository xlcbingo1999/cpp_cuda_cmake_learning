#include <iostream>
#include "inheritance.h"

Base::Base() {
    std::cout << "Base constructor" << std::endl;
}

Base::~Base() {
    std::cout << "Base destructor" << std::endl;
}

Base1::Base1() {
    std::cout << "Base1 constructor" << std::endl;
}

Base1::~Base1() {
    std::cout << "Base1 destructor" << std::endl;
}

Base2::Base2() {
    std::cout << "Base2 constructor" << std::endl;
}

Base2::~Base2() {
    std::cout << "Base2 destructor" << std::endl;
}

Base3::Base3() {
    std::cout << "Base3 constructor" << std::endl;
}

Base3::~Base3() {
    std::cout << "Base3 destructor" << std::endl;
}

MyClass::MyClass(int _num1, int _num2) {
    this->num1 = _num1;
    this->num2 = _num2;
    std::cout << "MyClass constructor" << std::endl;
}

MyClass::~MyClass() {
    std::cout << "MyClass destructor" << std::endl;
}

TempBase::TempBase(int _t) {
    this->temp = _t;
    std::cout << "TempBase constructor" << std::endl;
}

TempBase::~TempBase() {
    std::cout << "TempBase destructor" << std::endl;
}

void TempBase::print() {
    std::cout << "TempBase::temp: " << this->temp << std::endl;
}

TempBase1::TempBase1(int _t1, int _t2) : TempBase(_t1) {
    this->temp1 = _t2;
    std::cout << "TempBase1 constructor" << std::endl;
}

TempBase1::~TempBase1() {
    std::cout << "TempBase1 destructor" << std::endl;
}

void TempBase1::print() {
    TempBase::print();
    std::cout << "TempBase1::temp1: " << this->temp1 << std::endl;
}

TempBase2::TempBase2(int _t1, int _t2) : TempBase(_t1) {
    this->temp2 = _t2;
    std::cout << "TempBase2 constructor" << std::endl;
}

TempBase2::~TempBase2() {
    std::cout << "TempBase2 destructor" << std::endl;
}

void TempBase2::print() {
    TempBase::print();
    std::cout << "TempBase2::temp2: " << this->temp2 << std::endl;
}

// 继承的时候必须必须为基类写对应的初始化参数, 因为继承会把所有的成员变量都继承下来
MyTempClass::MyTempClass(int _t11, int _t21, int _t2, int _t3, int _t4) : TempBase1(_t11, _t2), TempBase2(_t21, _t3), temp3(_t4) {
    std::cout << "MyTempClass constructor" << std::endl;
}

MyTempClass::~MyTempClass() {
    std::cout << "MyTempClass destructor" << std::endl;
}

void MyTempClass::print() {
    TempBase1::print();
    TempBase2::print(); // 菱形继承中解决多继承问题的方法
    std::cout << "MyTempClass::temp3: " << this->temp3 << std::endl;

}
