#include <functional>
#include <vector>
#include <omp.h>
#include <cstdio>
#include "lock.h"


static omp_lock_t lock;

void lockFunc(int a, int b) {
    printf("func call a: %d, b: %d\n", a, b);

    omp_set_lock(&lock);
    printf("p BB1 a: %d, b: %d\n", a, b);
    printf("p BB2 a: %d, b: %d\n", a, b);
    omp_unset_lock(&lock);
}

int runLock() {
    const int task_groups = 5;

    std::vector<std::function<void(void)>> tasks; // 构建一个任务队列 句柄是传入void类型的参数, 返回void类型的返回值
    for (int i = 0; i < task_groups; i++) {
        tasks.emplace_back(std::bind(lockFunc, 10, i*10)); // 绑定成函数句柄, 然后发送到任务队列中
        tasks.emplace_back(std::bind([=] (void) {
            // 值捕获
            printf("lambda: i: %d\n", i);
        }));
    }

    size_t size = tasks.size();


    omp_init_lock(&lock);
    #pragma omp parallel for num_threads(1)
    for (size_t i = 0; i < size; i++) {
        tasks[i](); // 并行执行?
    }
    omp_destroy_lock(&lock);
    return 0;
}