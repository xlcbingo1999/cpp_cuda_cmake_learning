#ifndef __CMAKE_INTERFACE_SUB_H__
#define __CMAKE_INTERFACE_SUB_H__

#include "add.h"

namespace CMAKEINTERFACE {
    class Sub {
    private:
        int a;
        int b;
        CMAKEINTERFACE::Add* additem;

    public:
        Sub(int _a, int _b, CMAKEINTERFACE::Add* _additem);
        int sub();
        int subC(int c);
    };
};


#endif // __CMAKE_INTERFACE_SUB_H__