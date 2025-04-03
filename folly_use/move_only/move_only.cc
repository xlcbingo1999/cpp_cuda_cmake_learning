#include <iostream>
#include <folly/Utility.h>

class StorageClient : public folly::MoveOnly {
public:
    StorageClient() {
        std::cout << "StorageClient constructed" << std::endl;
    }
};

int main() {
    StorageClient a;
    StorageClient b = std::move(a);  // 移动构造
    // StorageClient c = a;  // 错误：StorageClient 是 MoveOnly 类型，不能复制
}