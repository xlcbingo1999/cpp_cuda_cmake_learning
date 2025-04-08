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
      return Chat{*this};
    }

    std::suspend_always initial_suspend() noexcept
    { // start up
      return {};
    }
    std::suspend_always
    yield_value(std::string msg) noexcept
    { // Value from co_yield
      _msgOut = std::move(msg);
      return {};
    }

    auto await_transform(std::string) noexcept
    { // Value from co_await
      struct awaiter
      { // Customized version instead of using suspend_never
        promise_type &pt;
        constexpr bool await_ready() const noexcept { return true; }

        std::string await_resume() const noexcept
        {
          return std::move(pt._msgIn);
        }
        void await_suspend(std::coroutine_handle<>) const noexcept {}
      };

      return awaiter{*this};
    }

    void return_value(std::string msg) noexcept
    { // value from co_return
      _msgOut = std::move(msg);
    }

    std::suspend_always final_suspend() noexcept
    { // ending
      return {};
    }
  };

  std::coroutine_handle<promise_type> mHandle{};

  explicit Chat(promise_type &p)
      : mHandle(std::coroutine_handle<promise_type>::from_promise(p)) {}

  // std::exchange的用法：
  // https://blog.csdn.net/qq_21438461/article/details/131273986
  Chat(Chat &&rhs) noexcept : mHandle(std::exchange(rhs.mHandle, nullptr)) {}

  ~Chat() noexcept
  {
    if (mHandle)
    {
      mHandle.destroy(); // 销毁掉coroutine_handle
    }
  }

  std::string listen()
  {
    if (not mHandle.done())
    { // 如果还没完成, 还需要等待其完成
      mHandle.resume();
    }
    return std::move(mHandle.promise()._msgOut);
  }

  void answer(std::string msg)
  {
    mHandle.promise()._msgIn = std::move(msg);
    if (not mHandle.done())
    {
      mHandle.resume();
    }
  }
};

Chat Fun()
{
  co_yield "Hello!\n"; // 该操作会调用 promise_type.yield_value,
                       // 将结果传送给coroutine的调用者
  std::cout
      << co_await std::string{}; // 该操作会调用 promise_type.await_transform,
                                 // 从coroutine外读取输入数据
  co_return "Here!\n";           // 该操作会调用 promise_type.return_value,
                                 // 将结果传送给coroutine的调用者
}

void Use()
{
  Chat marco = Fun();

  std::cout << marco.listen();
  marco.answer("Where are you?\n");
  std::cout << marco.listen();
}

int main() { Use(); }