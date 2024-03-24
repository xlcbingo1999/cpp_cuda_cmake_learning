#include <iostream>
#include "testover.h"

void A::func1(int i, int j) {
    std::cout << "A::func1() : " << i <<" " << j << std::endl;
}
void A::func2(int i) {
    std::cout << "A::func2() : " << i << std::endl;
}
void A::func3(int i) {
    std::cout << "A::func3(int) : " << i << std::endl;
}

void B::func1(double i) { // 隐藏(Hiding) 发生在多个类中的func1隐藏, 此时会根据调用的对象类型来决定调用哪个版本的函数
    std::cout << "B::func1() : " << i << std::endl;
}
void B::func3(int i) { // 重写(Override) 发生在多个类中的func3重写, 此时会根据调用的对象类型来决定调用哪个版本的函数
    std::cout << "B::func3(int) : " << i << std::endl;
}
void B::func3(double i) { // 重载(Overload) 发生在同一个类中的func3重载?, 此时会根据函数签名来进行处理
    std::cout << "B::func3(double) : " << i << std::endl;
}