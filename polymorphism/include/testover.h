
class A {
public:
    void func1(int i, int j);
    void func2(int i);
    virtual void func3(int i);
};

class B: public A {
public:
    void func1(double i);
    void func3(int i);
    void func3(double i);
};