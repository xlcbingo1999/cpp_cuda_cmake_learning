#include "file.h"
#include <fstream>
#include <iostream>

namespace RAII {
    File::File(const char* filename) {
        this->m_handle = std::ifstream(filename);
    }

    File::~File() {
        if (this->m_handle.is_open()) {
            std::cout << "File is closed" << std::endl;
            this->m_handle.close(); // 自动析构的时候就要关闭
        }
    }

    std::ifstream& File::getHandle() {
        return this->m_handle;
    }
};