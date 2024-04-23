#include "UThreadPool.h"

int main() {
    UThreadPool* pool = new UThreadPool();
    
    std::vector<std::future<int>> localFuture;
    std::chrono::system_clock::time_point ddl = std::chrono::system_clock::now() + std::chrono::seconds(20);
    
    for (int i = 0; i < 20; i++) {
        UTask task = UTask(i);
        localFuture.emplace_back(pool->commit(task));
    }

    for (auto& fut: localFuture) {
        const auto& status = fut.wait_until(ddl);
        switch (status) {
            case std::future_status::ready: 
                std::cout << "result: " << fut.get() << std::endl;
                break;
            case std::future_status::timeout:
                std::cout << "fut timeout" << std::endl;
                break;
            case std::future_status::deferred:
                std::cout << "fut deferred" << std::endl;
                break;
            default:
                std::cout << "fut unknowned" << std::endl;
                break;
        }
    }
}