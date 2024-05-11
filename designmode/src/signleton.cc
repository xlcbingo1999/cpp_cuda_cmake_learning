#include "signleton.h"

Signle* Signle::signleton_ = nullptr; // 初始化
std::mutex Signle::signleton_mutex_; // 需要初始化static的锁

Signle::Signle(const std::string value) {
    if (this->_val == "") {
        std::unique_lock<std::mutex> lk(this->val_mutex_);
        if (this->_val == "") {
            this->_val = value;
            printf("modify %s\n", value.c_str());
        }
    }
}


Signle* Signle::getInstance(const std::string& value) {
    if (signleton_ == nullptr) {
        std::unique_lock<std::mutex> lk(signleton_mutex_);
        if (signleton_ == nullptr) {
            signleton_ = new Signle(value);
            printf("do set signleton: %s\n", value.c_str());
        }
    }
    return signleton_;
}


std::string Signle::getValue() const {
    return this->_val;
}