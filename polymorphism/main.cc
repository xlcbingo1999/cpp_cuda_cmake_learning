#include <bits/stdc++.h>
#include "testover.h"
#include "inheritance.h"
#include "virtualmulti.h"
using namespace std;


int main() {
    B b;
    A* pa = &b;
    B* pb = &b;
    pa->func3(3);  // 重写，多态性，调用B的函数
    b.func3(10);   // 根据参数选择调用哪个函数，可能重写也可能隐藏，调用B的函数
    pb->func3(20); //根据参数选择调用哪个函数，可能重写也可能隐藏，调用B的函数

    // MyClass obj = MyClass(1, 2);

    // 普通菱形继承
    // MyTempClass objTemp = MyTempClass(1, 2, 3, 4, 5);
    // objTemp.print();

    // cout << "objTemp.temp1: " << objTemp.temp1 << endl;
    // cout << "objTemp.temp2: " << objTemp.temp2 << endl;
    // cout << "objTemp.TempBase1::temp: " << objTemp.TempBase1::temp << endl; // 解决二义性的方法: 指定对应的基类类型即可
    // cout << "objTemp.TempBase2::temp: " << objTemp.TempBase2::temp << endl; // 解决二义性的方法: 指定对应的基类类型即可

    // 虚菱形继承
    VirturalMulti::MyTempClass objTemp = VirturalMulti::MyTempClass(1, 2, 3, 4);
    cout << "objTemp.temp: " << objTemp.temp << endl;
    cout << "objTemp.temp1: " << objTemp.temp1 << endl;
    cout << "objTemp.temp2: " << objTemp.temp2 << endl;
    cout << "objTemp.temp3: " << objTemp.temp3 << endl;
}
