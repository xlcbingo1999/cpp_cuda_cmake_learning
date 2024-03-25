
namespace Polymorphism {
    class Empty { 

    };

    class X: Empty {
    public:
        int a;
        X(int _a);
    };

    class Y: X {
    public:
        double b;
        Y(int _a, double _b);
    };

    #pragma (push, 1) // 取消内存对齐!
    class UnAliseZ {
    public: 
        char k;
        int j; // 4(会和另一个int一起对齐)
        int d; // 4(会和另一个int一起对齐)
        double c; // 8
        UnAliseZ(char k, int _j, int _d, double _c);
    };
    #pragma (pop)

    class Z {
    public: 
        char k;
        int j; // 4(会和另一个int一起对齐)
        int d; // 4(会和另一个int一起对齐)
        double c; // 8
        Z(char k, int _j, int _d, double _c);
    };

    void f(X* p);
    void f();

    extern int global_var; // 先声明, 如果本文件中找不到定义, 就链接别的文件的时候获取定义
};

