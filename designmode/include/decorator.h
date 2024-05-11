// RAII风格
// 多用点shared_ptr
// 装饰器

#include <string>

class Component {
public:
    virtual ~Component() {
        printf("~Component()\n");
    };
    virtual std::string Operation() const = 0; // 纯虚函数
};

class RealComponent: public Component {
public:
    virtual ~RealComponent() {
        printf("~RealComponent()\n");
    }
    std::string Operation() const override {
        std::string res = "RealComponent";
        printf("Operation: %s\n", res.c_str());
        return res;
    }
};

class Decorator: public Component {
protected:
    Component* component_; // 基类指针用于包装对应的多态子类
public:
    virtual ~Decorator() {
        printf("~Decorator()\n");
    }
    Decorator(Component* component): component_{component} {}
    std::string Operation() const override {
        return "Decorator(" + this->component_->Operation() + ")";
    }
};