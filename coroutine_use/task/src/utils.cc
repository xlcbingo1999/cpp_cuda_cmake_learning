#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

inline char separator() {
    #ifdef _WIN32
      return '\\';
    #else
      return '/';
    #endif
}

const char* file_name(const char *path) {
    const char *file = path;
    while (*path) {
        if (*path++ == separator()) {
            file = path;
        }
    }
    return file;
}


void PrintTime() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;    

    std::cout << std::put_time(std::localtime(&in_time_t), "%T") 
        << "." << std::setfill('0') << std::setw(3) << ms.count();
}


void PrintThread() {
    std::cout << " [Thread-" << std::this_thread::get_id() << "] ";
}