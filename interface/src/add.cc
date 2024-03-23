#include "add.h"

namespace CMAKEINTERFACE {
    Add::Add(int _a, int _b) {
        this->a = _a;
        this->b = _b;
    }

    int Add::add() {
        return this->a + this->b;
    }

    int Add::addC(int c) {
        return this->a + this->b + c;
    }
};
