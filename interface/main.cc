#include <iostream>
#include "add.h"
#include "sub.h"

int main() {
    int a = 1;
    int b = 3;
    int c = 5;

    CMAKEINTERFACE::Add* item = new CMAKEINTERFACE::Add(a, b);
    std::cout << item->add() << std::endl;
    std::cout << item->addC(c) << std::endl;

    CMAKEINTERFACE::Sub* subitem = new CMAKEINTERFACE::Sub(a, b, item);
    std::cout << subitem->sub() << std::endl;
    std::cout << subitem->subC(c) << std::endl;
}