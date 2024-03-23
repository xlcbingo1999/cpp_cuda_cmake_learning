#include "main.h"
#include <iostream>

int helloworld() {
    return 33;
}

int main() {
    std::cout << hello() << std::endl;
    std::cout << world() << std::endl;
    std::cout << helloworld() << std::endl;
    return 0;
}