#include <functional>
#include <vector>
#include "threadpool.h"

void func(int a, int b) {
    printf("func call a: %d, b: %d\n", a, b);
}

void MyTemp::mp(float d) {
    this->m_c += d; 
    printf("MyTemp::mp call d: %f, this->m_c: %f\n", d, this->m_c);
}

int runThreadPool() {
    const int task_groups = 5;
    MyTemp temp[task_groups];

    std::vector<std::function<void(void)>> tasks; // 构建一个任务队列 句柄是传入void类型的参数, 返回void类型的返回值
    for (int i = 0; i < task_groups; i++) {
        tasks.emplace_back(std::bind(func, 10, i*10)); // 绑定成函数句柄, 然后发送到任务队列中
        tasks.emplace_back(std::bind(&MyTemp::mp, &temp[i], i * 2.0f));
        tasks.emplace_back(std::bind([=] (void) {
            // 值捕获
            printf("lambda: i: %d\n", i);
        }));
    }

    size_t size = tasks.size();

    // #pragma omp parallel for 后续会跟随一个for循环
    // omp的三种调度方式
    // 1. 静态调度 schedule(static, size). 编译的时候确定如何调度, size表示每次调度的迭代数量, 应该就是线程池的数量
    // 2. 动态调度 schedule(dynamic, size). 运行的时候动态决定如何调度, size表示每次调度的迭代数量, 应该就是线程池的数量
    // 3. 启发式调度 schedule(guided, size). 启发式决定调度方案, size表示每次分配的迭代次数的最小值, 不会小于size
    // #pragma omp parallel for schedule(guided, 1)
    #pragma omp parallel for num_threads(1)
    for (size_t i = 0; i < size; i++) {
        tasks[i](); // 并行执行?
    }
    return 0;
}