#ifndef CGRAPH_COBJECT_H
#define CGRAPH_COBJECT_H


class CObject {
public:
    explicit CObject() = default; // 表示默认初始化, 避免进行隐式转换构造
    virtual void init() {

    }
    virtual int run() = 0; // 纯虚函数, 必须实现
    virtual void destroy() {
        
    }
    virtual ~CObject() = default; // 虚析构, 避免父类指针析构的时候不调用子类的析构函数
};

#endif //CGRAPH_COBJECT_H