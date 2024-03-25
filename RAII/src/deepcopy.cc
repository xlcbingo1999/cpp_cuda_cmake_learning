#include "deepcopy.h"
#include <cstring>
#include <cstdio>

namespace RAII {
    DeepCopyClass::DeepCopyClass(const char* str) {
        data = new char[strlen(str) + 1]; // 多加一格用于保存'\0'
        strcpy(data, str); // 完整拷贝
        // memcpy(newstr, pod->addr+10, 9); // 从src指针的移动10格指针位置开始拷贝, 拷贝9个字符, 可能不包含'\0'

    }

    DeepCopyClass::DeepCopyClass(const DeepCopyClass& other) {
        data = new char[strlen(other.data) + 1];
        strcpy(data, other.data); 
    }

    DeepCopyClass& DeepCopyClass::operator=(const DeepCopyClass& other) {
        if (this == &other) {
            return *this;
        }

        if (data != nullptr) {
            delete[] data; // 因为会有先前值, 需要删除这段空间
        }
        data = new char[strlen(other.data) + 1];
        strcpy(data, other.data);
        return *this;
    }

    DeepCopyClass::~DeepCopyClass() {
        delete[] data;
    }


    void DeepCopyClass::print() const {
        printf("%s\n", data);
    }

    DeepCopyClass& DeepCopyClass::setstring(const char* str) {
        if (data != nullptr) {
            delete[] data;
        }

        data = new char[strlen(str) + 1];
        strcpy(data, str);
        return *this; // 指针的引用
    }
};