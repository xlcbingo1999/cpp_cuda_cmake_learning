#include <memory>
namespace RAII {
    class SomeData; // 前向声明
    void SomeAPI(const std::shared_ptr<SomeData>& ptr); // 需要将智能指针传递给外部函数
    

    // 这里是继承了一个模板类, 其中给出了一个方法: shared_from_this() 可以返回一个weak的this指针给别人用, 且不会降低原本的引用计数
    class SomeData : public std::enable_shared_from_this<SomeData> {
    public:
        static std::shared_ptr<SomeData> Create(int _v);
        void NeedCallSomeAPI();
        int GetVal() const;
        void SetVal(int _v);
    private:
        SomeData(int _v); // 构造函数
        int val;
    };
};