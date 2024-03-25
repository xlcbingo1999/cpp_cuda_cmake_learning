namespace RAII {
    void change(int&& right_value);
    void rightValueFunc();

    class Array {
    public:
        Array(int _s);
        Array(const Array& other); // 左值深拷贝, 可以传入右值, 右值在使用完之后就没有意义了
        Array& operator=(const Array& other);
        Array(Array&& other); // 移动右值拷贝函数, 直接传递一个右值进来
        
        virtual ~Array();

        void setData(const int& count, ...);
        void print() const;
    private:
        int* data;
        int size;
    };
};