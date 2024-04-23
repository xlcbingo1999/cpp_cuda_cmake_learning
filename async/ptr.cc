#include <iostream>
#include <vector>
#include <memory>

class Wight {

public:
    Wight() {
        std::cout << "normal" << std::endl;
        v.clear();

    }

    Wight(const Wight& other) {
        std::cout << "copy" << std::endl;
        v.clear();
        for (int item: other.v) {
            v.push_back(item);
        }
    }

    Wight& operator=(const Wight& other) {
        std::cout << "fuzhi" << std::endl;
        v.clear();
        for (int item: other.v) {
            v.push_back(item);
        }
        return *this;
    }

    Wight(Wight&& other) { // 右值无法被取地址, 右值是表达式结束之后不再存在的一个临时对象
        std::cout << "move" << std::endl;
        v.clear();
        for (int item: other.v) {
            v.push_back(item);
        }
        other.v.clear();
    }

    ~Wight() {
        v.clear();
    }

    std::vector<int> v;
};

class Book
{
    int mCount{1};
    std::string mName;

public:
    std::string &get_name()
    {
        return mName;
    }
    Book(std::string iName) : mName(iName) {}
    Book(Book &&iBook)
    {
        std::cout<< (this->mName) <<iBook.mName<<std::endl;
        swap(this->mName, iBook.mName);
        mCount = 3;
    }
    int get_count()
    {
        return mCount;
    }
};

int main() {
    {
        Wight originW;
        originW.v.push_back(1);
        std::unique_ptr<Wight> wptr = std::make_unique<Wight>(std::move(originW)); // 移动过来, 如果没有写移动构造, 原来内容不会空
        // auto wptr2 = wptr; // 会报错
        auto wptr2 = std::move(wptr); // 对将要销毁的对象转移给其他变量, 而不是做一次重新的拷贝, 可以提高性能
        std::cout << "result" << std::endl;
    }
    
    {
        Book b("Im");
        Book tb = std::move(b);
        std::cout << "old b name is " << b.get_name() << " count is " << b.get_count() << std::endl;
        std::cout << "\ntb name is " << tb.get_name() << " count is " << tb.get_count() << std::endl;
    }
}