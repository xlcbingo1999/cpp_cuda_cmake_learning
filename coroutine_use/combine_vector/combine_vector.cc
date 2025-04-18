#include <iostream>
#include <coroutine>
#include <iterator>
#include <utility>
#include <vector>

struct Generator {
    struct promise_type {
        int _val{};

        Generator get_return_object() {
            return Generator{*this};
        }

        std::suspend_never initial_suspend() noexcept {
            return {};
        }

        std::suspend_always final_suspend() noexcept {
            return {};
        }
        
        std::suspend_always yield_value(int v) {
            _val = v;
            return {};
        }

        void return_void() noexcept {}
        void unhandled_exception() noexcept {}
    };

    std::coroutine_handle<promise_type> mHandle{};
    
    explicit Generator(promise_type &p) noexcept: mHandle(std::coroutine_handle<promise_type>::from_promise(p)) {}
    
    Generator(Generator &&rhs) noexcept: mHandle(std::exchange(rhs.mHandle, nullptr)) {}

    Generator(Generator &) = delete;
    Generator &operator=(Generator &) = delete;


    ~Generator() noexcept {
        if (mHandle) {
            mHandle.destroy();
        }
    }

    int value() const {
        return mHandle.promise()._val;
    }

    bool finished() const {
        return mHandle.done();
    }

    void resume() {
        if (not finished()) {
            mHandle.resume();
        }
    }

    struct iterator {
        std::coroutine_handle<promise_type> mHandle{};
        
        bool operator==(std::default_sentinel_t) const {
            return mHandle.done();
        }

        iterator& operator++() {
            mHandle.resume();
            return *this;
        }

        const int operator*() const {
            return mHandle.promise()._val;
        }
    };

    iterator begin() {
        return iterator{mHandle};
    }

    std::default_sentinel_t end() {
        return {};
    }
};

Generator interleaved(std::vector<int> a, std::vector<int> b) {
    auto lamb = [](std::vector<int> &v) -> Generator {
        for (const auto item: v) {
            co_yield item;
        }
    };

    auto x = lamb(a);
    auto y = lamb(b);

    while (not x.finished() or not y.finished()) {
        if (not x.finished()) {
            co_yield x.value();
            x.resume();
        }

        if (not y.finished()) {
            co_yield y.value();
            y.resume();
        }
    }
}

int main() {
    std::vector a{2, 4, 6, 8};
    std::vector b{1,3,5,7};

    Generator g{interleaved(std::move(a), std::move(b))};
    
    for (const auto item: g) {
        std::cout << item << std::endl;
    }

    // while (not g.finished()) {
    //     std::cout << g.value() << std::endl;
    //     g.resume();
    // }

    return 0;
}