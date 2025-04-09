#ifndef COROUTINEUSE_DISPATCHER_UTILS_H_
#define COROUTINEUSE_DISPATCHER_UTILS_H_


#include <iostream>

const char* file_name(const char *path);
void PrintTime();
void PrintThread();

template <typename ...U>
void Println(U... u) {
    int i = 0;
    auto printer = [&i]<typename Arg>(Arg arg) {
        if (sizeof...(u) == ++i) {
            std::cout << arg << std::endl;
        } else {
            std::cout << arg << " ";
        }
    };
    (printer(u), ...);

    std::cout.flush();
}

#define debug(...) \
    PrintTime();   \
    PrintThread(); \
    printf("(%s:%d) %s: ", file_name(__FILE__), __LINE__, __func__); \
    Println(__VA_ARGS__);

#endif //COROUTINEUSE_DISPATCHER_UTILS_H_