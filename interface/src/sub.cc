#include "sub.h"
#include "add.h"

namespace CMAKEINTERFACE {
    Sub::Sub(int _a, int _b, CMAKEINTERFACE::Add* _additem) {
        this->a = _a;
        this->b = _b;
        this->additem = _additem;
    }

    int Sub::sub() {
        return this->additem->add();
    }

    int Sub::subC(int c) {
        return this->additem->addC(-c);
    }
};
