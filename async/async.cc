#include <iostream>
#include <thread>
#include <string>
#include <future>
#include <chrono>
#include <functional>

std::string fetchDB(std::string recvData) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return "DB_" + recvData;
}

int main() {
    auto start = std::chrono::system_clock::now();

    // 第一种方案 函数
    std::future<std::string> resultF1 = std::async(
        std::launch::async, fetchDB, "data1"
    );


    // 第二种方案 函数时编程 bind
    std::future<std::string> resultF2 = std::async(
        std::launch::deferred, std::bind(fetchDB, "data2")
    );

    // 第三种方案 lambda函数
    std::string recvData3 = "data3";
    std::future<std::string> resultF3 = std::async(
        std::launch::deferred, [recvData3] () -> std::string {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            return "DB_" + recvData3;
        }
    );

    std::cout << "begin zuse" << std::endl;

    std::string res3 = "null";

    std::this_thread::sleep_for(std::chrono::seconds(2));
    const auto& fstatus = resultF3.wait_for(std::chrono::milliseconds(2));    
    switch(fstatus) {
        case std::future_status::ready: 
            std::cout << "ready" << std::endl;
            res3 = resultF3.get();
            break;
        case std::future_status::timeout:
            std::cout << "timeout" << std::endl;
            res3 = "timeout";
            break;
        // 传入第二个参数,std::launch::deferred，那么线程的执行就会推迟到std::future.get()方法时才会启动 如果不使用get或者wait时，线程直接结束
        case std::future_status::deferred:
            std::cout << "deferred" << std::endl;
            res3 = "deferred";
            break;
        default:
            std::cout << "nothing happen" << std::endl;
    }

    std::string res1 = resultF1.get(); // feature新建的时候就执行了
    std::string res2 = resultF2.get(); // .get()调用的时候才会执行deferred异步
    
    std::cout << "end zuse" << std::endl;

    auto end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
    std::cout << "total time: " << diff << std::endl;

    std::cout << "res1: " << res1 << " res2: " << res2 << "res3: " << res3 << std::endl;
}
