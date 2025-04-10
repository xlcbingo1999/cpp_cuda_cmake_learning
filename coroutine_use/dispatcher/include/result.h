#ifndef COROUTINEUSE_DISPATCHER_RESULT_H_
#define COROUTINEUSE_DISPATCHER_RESULT_H_


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


template <>
struct Result<void> {
    explicit Result() = default;
    explicit Result(std::exception_ptr &&rhs): _exception_ptr(rhs) {}

    void get_or_throw() {
        if (_exception_ptr) {
            std::rethrow_exception(_exception_ptr);
        }
    }
private:
    std::exception_ptr _exception_ptr;
};

#endif //COROUTINEUSE_DISPATCHER_RESULT_H_