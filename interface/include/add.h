#ifndef __CMAKE_INTERFACE_ADD_H__
#define __CMAKE_INTERFACE_ADD_H__

namespace CMAKEINTERFACE {
    class Add {
    private:
        int a;
        int b;

    public:
        Add(int _a, int _b);
        int add();
        int addC(int c);
    };
};


#endif // __CMAKE_INTERFACE_ADD_H__