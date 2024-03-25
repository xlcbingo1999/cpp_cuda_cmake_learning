#include "sizeof.h"
#include "csizeof.h"
#include <iostream>
#include <arpa/inet.h>
using namespace std;


namespace Polymorphism {
    X::X(int _a) {
        this->a = _a;
    }

    Y::Y(int _a, double _b) : X(_a) {
        this->b = _b;
    }

    UnAliseZ::UnAliseZ(char _k, int _j, int _d, double _c) {
        this->k = _k;
        this->j = _j;
        this->d = _d;
        this->c = _c;
    }

    Z::Z(char _k, int _j, int _d, double _c) {
        this->k = _k;
        this->j = _j;
        this->d = _d;
        this->c = _c;
    }

    void f() {
        Empty a, b;
        cout << "sizeof(a): " << sizeof(a) << "; sizeof(b): " << sizeof(b) << endl;
        if (&a == &b) {
            cout << "impossible: report error to compiler supplier";
        }

        Empty* p1 = new Empty;
        Empty* p2 = new Empty;
        // 指针类型的sizeof是根据计算机位数来决定的，32位是4，64位是8
        cout << "sizeof(p1): " << sizeof(p1) << "; sizeof(p2): " << sizeof(p2) << endl;
        cout << "sizeof(*p1): " << sizeof(*p1) << "; sizeof(*p2): " << sizeof(*p2) << endl;

        if (p1 == p2) {
            cout << "impossible: report error to compiler supplier";
        }

        X* x1 = new X(12);
        cout << "sizeof(*x1): " << sizeof(*x1) << endl;

        Y* y1 = new Y(12, 0.2);
        cout << "sizeof(*y1): " << sizeof(*y1) << endl; 

        Z* z1 = new Z('1', 2, 1, 0.3);
        cout << "sizeof(*z1): " << sizeof(*z1) << endl;

        UnAliseZ* z2 = new UnAliseZ('1', 2, 1, 0.3);
        cout << "sizeof(*z2): " << sizeof(*z2) << endl;

        {
            int smallendnum = 0x00636261;
            {
                // reinterpret_cast内容
                int *smallendptr = &smallendnum;
                // reinterpret_cast可以将一个对象转化为char*字符流
                char* ptr = reinterpret_cast<char*>(smallendptr);
                
                cout << "*smallendptr: " << *smallendptr << endl;
                // cout << "*ptr: " << hex << *ptr << endl;
                cout << "ptr: " << ptr << endl; // 直接使用ptr就可以获取其指向的内容, 而且会转化为char
                cout << "static_cast<void*> ptr: " << static_cast<void*>(ptr) << endl;
            }
            
            {
                char* smallendnumchar = (char*)&smallendnum;
                if (*smallendnumchar) {
                    cout << "little end" << endl;
                } else {
                    cout << "small end" << endl;
                }
                printf("%x %x %x %x\n", smallendnumchar[0], smallendnumchar[1], smallendnumchar[2], smallendnumchar[3]);
            }

            {
                int bigendnum = ((smallendnum>>24)&0xff) |
                                ((smallendnum<<8)&0xff0000) |
                                ((smallendnum>>8)&0xff00) |
                                ((smallendnum<<24)&0xff000000);
                printf("%x\n", bigendnum);
            }
        }

        {
            unsigned int hostValue = 0x11223344;
            unsigned int networkValue = htonl(hostValue);

            printf("Host value: 0x%08x\n", hostValue);
            printf("Network value: 0x%08x\n", networkValue);

            unsigned int rehostValue = ntohl(networkValue);
            printf("Re Host value: 0x%08x\n", rehostValue);
        }

        {
            cout << "global_var: " << global_var << endl;
        }
        

        // {
        //     print_message("Hello, world!");
        // }
    }

    void f(X* p) {
        void* pa = &(p->a);
        void* pb = p;
        if (pa == pb) {
            cout << "nice" << endl;
        }
    }
};


