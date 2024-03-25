namespace RAII {
    class DeepCopyClass {
    private:
        // 如果没有实现深拷贝, C++默认只会进行浅拷贝, 指针成员只会把指针的值复制过去, 而不会开一个新的内存
        // 此时析构掉一个实例, 但是另一个实例还是会指向这块内存区域, 导致内存泄漏的产生
        char* data;
    public:
        DeepCopyClass(const char* str); // 常量的指针作为参数, 一般来说自己手写的字符串就是一个const char类型
        DeepCopyClass(const DeepCopyClass& other); // 常量的引用作为参数, 表示引用的对象是一个常量, 不能修改
        DeepCopyClass& operator=(const DeepCopyClass& other);
        ~DeepCopyClass();
        void print() const; // const修饰成员函数, 不能修改对象的状态
        DeepCopyClass& setstring(const char* str);
    };

};