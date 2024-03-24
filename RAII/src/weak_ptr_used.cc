#include "weak_ptr_used.h"
#include <iostream>

namespace RAII {
    PtrA::~PtrA() {
        std::cout << "PtrA dec" << std::endl;
    }

    PtrB::~PtrB() {
        std::cout << "PtrB dec" << std::endl;
    }
};