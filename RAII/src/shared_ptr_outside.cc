#include "shared_ptr_outside.h"
#include <iostream>

namespace RAII {
    void SomeAPI(const std::shared_ptr<SomeData>& ptr) {
        std::cout << "GetVal(1): " << ptr->GetVal() << std::endl;
        ptr->SetVal(20);
        std::cout << "GetVal(2): " << ptr->GetVal() << std::endl;
    }

    SomeData::SomeData(int _v) : val(_v) {}

    std::shared_ptr<SomeData> SomeData::Create(int _v) {
        return std::shared_ptr<SomeData>(new SomeData(_v)); // 调用构造函数创建一个新的智能指针
    }

    void SomeData::NeedCallSomeAPI() {
        SomeAPI(shared_from_this());
    }

    int SomeData::GetVal() const {
        return val;
    }

    void SomeData::SetVal(int _v) {
        val = _v;
    }
};
