#include "shared_ptr_used.h"
#include <iostream>


namespace RAII {
    SharedPtrUsed::SharedPtrUsed() {
        std::cout << "SharedPtrUsed con" << std::endl;
    }
    SharedPtrUsed::~SharedPtrUsed() {
        std::cout << "SharedPtrUsed dec" << std::endl;
    }
    void SharedPtrUsed::doSth() {
        std::cout << "SharedPtrUsed dosth" << std::endl;
    }
};