#include <coroutine>
#include <iostream>
#include <string>
#include <utility>

struct Chat
{
    struct promise_type
    {
        std::string _msgOut{}, _msgIn{};       // Storing a value from or for the
                                               // coroutine
        void unhandled_exception() noexcept {} // What to do in case of an exception
        Chat get_return_object()
        { // Coroutine creation
            std::cout << "In Chat get_return_object()" << std::endl;
            return Chat{*this};
        }

        std::suspend_always initial_suspend() noexcept
        { // start up
            std::cout << "In std::suspend_always initial_suspend()" << std::endl;
            return {};
        }

        // co_yield = co_await promise.yield_value(expr)
        // 因此实现这个函数之后，就可以让 co_yield 进行相关的处理了！
        // 这个函数专门被 co_yield 被用来向 coroutine 外面传递数据的
        // 通常情况下我们使用 co_await 更多的关注点在挂起自己，等待别人上，而使用 co_yield 则是挂起自己传值出去。
        std::suspend_always yield_value(std::string msg) noexcept
        { // Value from co_yield
            std::cout << "In std::suspend_always yield_value(std::string " << msg << ")" << std::endl;
            _msgOut = std::move(msg);
            return {};
        }

        // 这个函数是为了处理调用coroutine时，co_await之后跟着的不是awaiter结构体的场景
        // coroutine设计了这个函数，只需要重写这个函数，就可以在 co_await 之后跟上
        // std::string 类型的数据
        auto await_transform(std::string) noexcept
        { // Value from co_await
            std::cout << "In auto await_transform(std::string" << ")" << std::endl;
            struct awaiter
            { // Customized version instead of using suspend_never
                promise_type &pt;
                // await_ready 返回 true 时候表示coroutine从不挂起，返回 false
                // 时候表示coroutine会被挂起
                constexpr bool await_ready() const noexcept { return true; }

                // coroutine 恢复执行之后，等待体的 await_resume 函数被调用。
                // await_resume 的返回值就是 co_await 的返回值
                std::string await_resume() const noexcept
                {
                    std::cout << "triggle std::string await_resume" << std::endl;
                    return std::move(pt._msgIn);
                }

                // 当await_ready返回false时，说明co_await调用会让coroutine进入挂起状态，此时就会进入到await_suspend函数中
                // 由于在本设计中，co_await调用不会挂起，因此这里就没有做相应的实现
                // await_suspend的返回值也是有门道的：
                // - 返回 void 类型或者返回
                // true，表示当前协程挂起之后将执行权还给当初调用或者恢复当前协程的函数。
                // - 返回 false，则恢复执行当前协程。注意此时不同于 await_ready 返回
                // true 的情形，此时协程已经挂起，await_suspend 返回 false
                // 相当于挂起又立即恢复。
                // - 返回其他协程的 coroutine_handle 对象，这时候返回的 coroutine_handle
                // 对应的协程被恢复执行。
                // - 抛出异常，此时当前协程恢复执行，并在当前协程当中抛出异常。
                void await_suspend(std::coroutine_handle<>) const noexcept {
                    std::cout << "triggle std::string await_suspend" << std::endl;
                }
            };

            return awaiter{*this};
        }

        void return_value(std::string msg) noexcept
        { // value from co_return
            std::cout << "In void return_value(std::string " << msg << ")" << std::endl;
            _msgOut = std::move(msg);
        }

        // 一般而言，这个final_suspend()需要返回 std::suspend_always，因为要让
        // coroutine 的生命周期延长，让其由Chat的析构函数来释放
        // final_suspend()是coroutine在最后一次执行完之后确定是否挂起的操作，这里默认都将其挂起
        std::suspend_always final_suspend() noexcept
        { // ending
            std::cout << "In std::suspend_always final_suspend(" << ")" << std::endl;
            return {};
        }
    };

    std::coroutine_handle<promise_type> mHandle{};

    // Chat 应该只支持以下几种构造方式：
    // - 传入promise_type的普通构造函数
    // -
    // 传入Chat右值的异构构造函数，保证对每一个coroutine实例，只有一个Chat对象存活
    // Chat 禁止的构造方式：
    // - 拷贝构造

    explicit Chat(promise_type &p)
        : mHandle(std::coroutine_handle<promise_type>::from_promise(p)) {}

    // std::exchange的用法：
    // https://blog.csdn.net/qq_21438461/article/details/131273986
    Chat(Chat &&rhs) noexcept : mHandle(std::exchange(rhs.mHandle, nullptr)) {}

    Chat(Chat &) = delete;
    Chat &operator=(Chat &) = delete;

    // coroutine最后一次执行完之后，需要由Chat的析构函数来释放coroutine的内存，使用显式的destroy()函数
    // 让协程的状态的生成周期与 Chat 一致
    ~Chat() noexcept
    {
        std::cout << "destroy mHandle" << std::endl;
        if (mHandle)
        {
            mHandle.destroy(); // 销毁掉coroutine_handle
        }
    }

    std::string listen()
    {
        std::cout << "In std::string listen(" << ")" << " triggle mHandle.resume() " << std::endl;
        if (not mHandle.done())
        { // 如果还没完成, 还需要等待其完成
            mHandle.resume();
        }
        std::cout << "In std::string listen(" << ")" << " after mHandle.resume() " << std::endl;
        return std::move(mHandle.promise()._msgOut);
    }

    void answer(std::string msg)
    {
        mHandle.promise()._msgIn = std::move(msg);
        std::cout << "In std::string answer(" << ")" << " triggle mHandle.resume() " << std::endl;
        if (not mHandle.done())
        {
            mHandle.resume();
        }
        std::cout << "In std::string answer(" << ")" << " after mHandle.resume() " << std::endl;
    }
};

Chat Fun()
{
    std::cout << "In Chat Fun()" << " triggle co_yield Hello " << std::endl;
    co_yield "Hello!\n"; // 该操作会调用 promise_type.yield_value,
                         // 将结果传送给coroutine的调用者
    std::cout << "In Chat Fun()" << " after co_yield Hello " << std::endl;
    
    std::cout << "In Chat Fun()" << " triggle co_await What happen? " << std::endl;
    std::cout << co_await "What happen?"; // 该操作会调用 promise_type.await_transform,
                                   // 从coroutine外读取输入数据
    std::cout << "In Chat Fun()" << " after co_await What happen? " << std::endl;

    std::cout << "In Chat Fun()" << " triggle co_return Hello! " << std::endl;
    co_return "Here!\n";           // 该操作会调用 promise_type.return_value,
                                   // 将结果传送给coroutine的调用者
    std::cout << "In Chat Fun()" << " after co_return Hello! " << std::endl;
}

void Use()
{
    std::cout << "In Chat Fun()" << " triggle Chat marco = Fun() " << std::endl;
    Chat marco = Fun();
    std::cout << "In Chat Fun()" << " after Chat marco = Fun() " << std::endl;

    std::cout << "In Chat Fun()" << " triggle marco.listen()" << std::endl;
    std::cout << marco.listen();
    std::cout << "In Chat Fun()" << " after marco.listen()" << std::endl;

    std::cout << "In Chat Fun()" << " triggle marco.answer()" << std::endl;
    marco.answer("Where are you?\n");
    std::cout << "In Chat Fun()" << " after marco.answer()" << std::endl;

    std::cout << "In Chat Fun()" << " triggle marco.listen()" << std::endl;
    std::cout << marco.listen();
    std::cout << "In Chat Fun()" << " after marco.listen()" << std::endl;
}

int main() { Use(); }