
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
        // 避免出现循环引用, 这种用法经常在OC和Swift中使用, 不会增加引用计数！！！
        // 参考文献: https://csguide.cn/cpp/memory/how_to_understand_weak_ptr.html#weak-ptr-%E6%98%AF%E4%BB%80%E4%B9%88
        // weak_ptr引用的对象是弱引用关系, 它对对象并没有所有权
        // 1. 一个公司类可以拥有员工，那么这些员工就使用std::shared_ptr维护。
        //    另外有时候我们希望员工也能找到他的公司，所以也是用std::shared_ptr维护，这个时候问题就出来了。
        //    但是实际情况是，员工并不拥有公司，所以应该用std::weak_ptr来维护对公司的指针。

        // 2. 我们要使用异步方式执行一系列的Task，并且Task执行完毕后获取最后的结果。所以发起Task的一方和异步执行Task的一方都需要拥有Task。
        //    但是有时候，我们还想去了解一个Task的执行状态，比如每10秒看看进度如何，这种时候也许我们会将Task放到一个链表中做监控。
        //    这里需要注意的是，这个监控链表并不应该拥有Task本身，放到链表中的Task的生命周期不应该被一个观察者修改。所以这个时候就需要用到std::weak_ptr来安全的访问Task对象了。

        std::weak_ptr<PtrA> a_ptr;
        // std::shared_ptr<PtrA> a_ptr;
        ~PtrB();
    };
};