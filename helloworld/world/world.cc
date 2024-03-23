#include "world.h"
// #include "hello.h" // 在PRIVATE传递下, 如果是放在这里就不会报错了

int world() {
    return hello() + 1; // INTERFACE下会报错
    // return 1;
}