#ifndef COROUTINEUSE_TASK_RESULT_H_
#define COROUTINEUSE_TASK_RESULT_H_


#include <exception>

template <typename T>
struct Result {
    explicit Result() = default;
    explicit Result(T &&rhs): _value(rhs) {}
    explicit Result(std::exception_ptr &&rhs): _exception_ptr(rhs) {}

    T get_or_throw() {
        if (_exception_ptr) {
            std::rethrow_exception(_exception_ptr);
        }
        return _value;
    }
private:
    T _value{};
    std::exception_ptr _exception_ptr;
};

#endif //COROUTINEUSE_TASK_RESULT_H_