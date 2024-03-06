#include <iostream>

#include "pch.h"

#define GLFW_INCLUDE_NODE
#include "GLFW/glfw3.h"

#include "toml++/toml.h"

#include "core/log.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"

class Date {
private:
    int year;
    int month;
    int day;

public:
    Date(int _year, int _month, int _day): year(_year), month(_month), day(_day) {}
    friend std::ostream& operator<<(std::ostream& out, const Date& item) {
        out << item.year << "-" << item.month << "-" << item.day;
        return out;
    }
};



int main() {
    LOGI("xlc test begin [IN engine]");
    LOGI("mono: ", MONO_PATH);
    auto mono = mono_jit_init("my mono");
    mono_jit_cleanup(mono);

    
    Date date(2000, 11, 30);
    std::cout << date << std::endl;
    return 0;
}