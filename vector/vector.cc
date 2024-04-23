#include <iostream>
#include <vector>
#include <cstring>

class MyClass {
private:
    char* val;
    int id;
public:
    MyClass(const char* v, int id) {
        int len = strlen(v);
        this->val = new char[len+1];
        strcpy(this->val, v);
        this->val[len] = '\0';

        this->id = id;
        std::cout << "call MyClass(const char* v, int id)" << std::endl;
    }
    

    MyClass(const MyClass& other) {
        // this->val = other.val;
        int len = strlen(other.val);
        this->val = new char[len+1];
        strcpy(this->val, other.val);
        this->val[len] = '\0';

        this->id = other.id;
        std::cout << "call MyClass(const MyClass& other)" << std::endl;
    }

    MyClass& operator=(const MyClass& other) {
        // this->val = other.val;
        int len = strlen(other.val);
        this->val = new char[len+1];
        strcpy(this->val, other.val);
        this->val[len] = '\0';

        this->id = other.id;
        std::cout << "call MyClass& operator=(const MyClass& other)" << std::endl;
        return *this;
    }

    MyClass(MyClass&& other) {
        this->val = other.val;
        this->id = other.id;
        other.val = nullptr; // 直接赋值为nullptr即可
        std::cout << "call MyClass(MyClass&& other)" << std::endl;
    }
};

int main() {
    std::vector<MyClass> vec;
    std::vector<MyClass> vec1;
    std::vector<MyClass> vec2;

    vec.push_back(MyClass("sadsada", 1));
    std::cout << "==========1============" << std::endl;


    vec1.emplace_back("sadasdad", 2);
    std::cout << "==========2============" << std::endl;


    vec2.emplace_back(MyClass("sadadsadad", 3));
    std::cout << "==========3============" << std::endl;

}