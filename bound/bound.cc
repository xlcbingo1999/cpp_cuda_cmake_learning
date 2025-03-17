#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    std::vector<double> double_vec = {1.2, 2.3, 3.4, 5.0, 7.8};
    
    for (int i = 0; i < 8; i++) {
        auto lower_iter = std::lower_bound(double_vec.begin(), double_vec.end(), static_cast<double>(i));
        if (lower_iter != double_vec.end()) {
            std::cout << "i: " << i << " find lower_iter " << *lower_iter << std::endl;
        } else {
            std::cout << "i: " << i << " no find" << std::endl;
        }

        auto upper_iter = std::upper_bound(double_vec.begin(), double_vec.end(), static_cast<double>(i));
        if (upper_iter != double_vec.end()) {
            std::cout << "i: " << i << " find upper_iter " << *upper_iter << std::endl;
        } else {
            std::cout << "i: " << i << " no find" << std::endl;
        }
    }
}