
#include <memory>

namespace RAII {
    class PtrB;
    class PtrA {
    public:
        std::shared_ptr<PtrB> b_ptr;
        ~PtrA();
    };

    class PtrB {
    public:
        // 避免出现循环引用, 这种用法经常在OC和Swift中使用
        std::weak_ptr<PtrA> a_ptr;
        // std::shared_ptr<PtrA> a_ptr;
        ~PtrB();
    };
};