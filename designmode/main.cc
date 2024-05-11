#include "signleton.h"
#include "decorator.h"
#include <thread>
#include <vector>

int main() {
    {
        std::vector<std::thread> threadpool;
        int threadCount = 20;
        for (int i = 0; i < threadCount; i++) {
            threadpool.emplace_back([=] () {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                std::string str = std::to_string(i);
                Signle* sign = Signle::getInstance(str);
                printf("%i: %s\n", i , sign->getValue().c_str());
            });
        }
        
        for (int i = 0; i < threadCount; i++) {
            threadpool[i].join();
        }

        printf("all finished\n");
    }

    {
        std::shared_ptr<RealComponent> ptrRealComponent = std::make_shared<RealComponent>();
        // 暂时需要用原指针进行拷贝构造?
        std::shared_ptr<Decorator> ptrDecorator = std::make_shared<Decorator>(ptrRealComponent.get());

        std::string res = ptrDecorator->Operation();
        printf("res Operation: %s\n", res.c_str());
    }
}