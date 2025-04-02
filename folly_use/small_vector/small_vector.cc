#include <folly/small_vector.h>
#include <iostream>

int main() {
    // 定义一个最多在栈上存储4个元素的small_vector
    folly::small_vector<int, 4> vec;

    // 添加元素（前4个在栈上，超过后自动转到堆）
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    vec.emplace_back(4);  // 栈上存储
    vec.emplace_back(5);  // 超出容量，触发堆分配 [[1]][[5]]

    // 遍历输出
    std::cout << "Elements in small_vector: ";
    for (const auto& num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}