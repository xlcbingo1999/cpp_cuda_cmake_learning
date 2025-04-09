#include <functional>
#include <iostream>
#include <coroutine>
#include <list>
#include <utility>

template <typename T>
struct Generator {

    class ExhaustedException : std::exception {};

    struct promise_type {
        // 此时不需要用容器来做任何的存储，只需要每次存储一个值，然后coroutine的挂起和恢复时进行相应的操作即可
        T _value{};
        // 在coroutine执行的过程中，需要一个is_ready来标识目前操作已经做完了
        bool _is_ready; 

        Generator get_return_object() {
            return Generator{*this};
        }

        std::suspend_never initial_suspend() noexcept {
            return {};
        }

        std::suspend_always final_suspend() noexcept {
            return {};
        }

        std::suspend_always yield_value(T v) noexcept {
            _value = v;
            _is_ready = true; // 生产出来value之后就要设置ready标识
            return {};
        }

        void return_void() noexcept {
            
        }

        void unhandled_exception() noexcept {

        }
    };

    std::coroutine_handle<promise_type> mHandle{};
    
    Generator(promise_type &p) noexcept: mHandle(std::coroutine_handle<promise_type>::from_promise(p)) {}
    Generator(Generator &&rhs) noexcept: mHandle(std::exchange(rhs.mHandle, nullptr)) {}
    Generator(Generator &) = delete;
    Generator& operator=(Generator &) = delete;
    ~Generator() noexcept {
        if (mHandle) {
            mHandle.destroy();
        }
    }

    bool has_next() {
        if (!mHandle || mHandle.done()) {
            return false;
        }

        if (!mHandle.promise()._is_ready) { // 这里就是生产者，如果此时没有任何数值被生产出来，就需要让渡执行权给coroutine进行生产
            mHandle.resume();
        }

        if (mHandle.done()) { // 因为可能coroutine重新执行完之后导致状态改变了，因此需要在这里再加一次判断 
            return false;
        } else {
            return true;
        }
    }

    T next() {
        if (has_next()) {
            mHandle.promise()._is_ready = false; // 这是一个消费者，当消费了数字后就要设置状态
            return mHandle.promise()._value;
        }
        throw ExhaustedException();
    }

    template <typename F>
    void for_each(F f) { // 遍历是成员函数
        while (has_next()) {
            f(next());
        }
    }

    T sum() {
        T sum = 0;
        while (has_next()) {
            sum += next();
        }
        return sum;
    }

    template <typename R, typename F>
    R reduce(R init, F f) {
        R result = init;
        while (has_next()) {
            result = f(result, next());
        }
        return result;
    }

    template <typename F>
    Generator filter(F f) { // 这里要返回 Generator 对象，是因为结果是要下次才能用的，且相应的生产规则会进行改变，需要改变co_yield方式
        Generator g = std::move(*this); // 只能移动，不能拷贝
        while (g.has_next()) {
            T origin = g.next();
            if (f(origin)) {
                co_yield origin;
            }
        }
        // 这里似乎是不需要返回g的
    }

    template <typename F>
    Generator map(std::function<F(T)> f) {
        Generator g = std::move(*this);
        while (g.has_next()) {
            co_yield f(g.next());
        }
    }

    template <typename F>
    Generator<std::invoke_result_t<F, T>> map(F f) {
        Generator g = std::move(*this);
        while (g.has_next()) {
            co_yield f(g.next());
        }
    }

    template <typename F>
    std::invoke_result_t<F, T> flat_map(F f) {
        Generator g = std::move(*this);
        while (g.has_next()) {
            Generator new_g = f(g.next()); // 值会映射为新的Generator

            while (new_g.has_next()) { // 展开新的Generator，每个小的Generator都会自己处理一个seq逻辑
                co_yield new_g.next();
            }
        }
    }

    template <typename F>
    Generator take_while(F f) {
        Generator g = std::move(*this);
        while (g.has_next()) {
            auto nex = g.next();
            
            if (f(nex)) {
                co_yield nex;
            } else {
                break; // 当已经结束条件，就不需要进行了 
            }
        }
    }


    static Generator from_list(std::list<T> l) {
        for (const auto item: l) {
            co_yield item; // 每一次让渡给coroutine的操作都是让coroutine从这里重新执行
        }
    }

    static Generator from(std::initializer_list<T> l) {
        for (const auto item: l) {
            co_yield item;
        }
    }

    // 折叠表达式（fold expression）  
    template <typename ...Targs>
    static Generator from(Targs ...args) {
        (co_yield args, ...);
    }
};


int main()
{
    Generator<int>::from_list(std::list<int>{1, 2, 3, 4, 5})
        .for_each([](auto i) -> void {
            std::cout << i << std::endl;
        });

    std::cout << "test sum: " << Generator<int>::from_list({2, 4, 6}).sum() << std::endl;

    std::cout << "test reduce: "  << Generator<int>::from_list({2, 4, 6}).reduce(1, [](auto base, auto i) -> int {
        return base * i;
    }) << std::endl;


    Generator<float>::from(
        {2, 4, 6}
    ).filter([](auto i) -> bool {
        if (i > 2) {
            return true;
        }
        return false;
    }).for_each([] (auto i) -> void {
       std::cout << i << std::endl; 
    });


    Generator<float>::from(
        {2, 4, 6}
    ).map<float>([](auto i) -> float {
        return i * 4;
    }).for_each([] (auto i) -> void {
       std::cout << i << std::endl; 
    });

    Generator<float>::from(
        {4, 8, 12}
    ).map([](auto i) -> float {
        return i * 4;
    }).for_each([] (auto i) -> void {
       std::cout << i << std::endl; 
    });

    Generator<int>::from(
        0, 1, 2, 3, 4, 5
    ).flat_map([](auto i) -> Generator<int> {
        for (int j = 0; j < i; j++) {
            co_yield j;
        }
    }).for_each([] (auto i) -> void {
        if (i == 0) {
            std::cout << std::endl;
        } else {
            std::cout << "* ";
        }
    });


    Generator<float>::from(
        {4, 8, 12}
    ).take_while([](auto i) -> float {
        return i <= 8;
    }).for_each([] (auto i) -> void {
       std::cout << i << std::endl; 
    });
}