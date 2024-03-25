#include <vector>

namespace Polymorphism {
    typedef std::vector<int> IntVec; // 编译时执行, 支持严格的类型检查

    template <typename T> // 支持模板
    struct MyContainer {
        typedef std::vector<T> Type;
    };

    class Base {
    public:
        virtual void print();
    };

    class Deride: public Base {
    public:
        void print();
    };

    void RunRTTI();

};

#include "../src/rtti.inl"