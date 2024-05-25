#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void threadBlockGridTest(void) {
    // 在这种场景中, 每个thread都会完整执行整个操作

    
    // thread => 一个thread由GPU的一个核进行处理
        // (threadIdx.x/y/z是block中当前的thread所在的坐标位置)
    // block => 多个thread组成一个block, block间无法通信, 所有的block是并行执行的 
        // (blockDim.x/y/z是block中的各个维度的大小, blockIdx.x/y/z是grid中当前的block所在的坐标位置)
    // grid => 多个block组成一个grid
        // (gridDim.x/y/z是grid中各个维度的大小)

    // 获取一个grid中包含多少个总的thread
    int allThreadNums = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    printf("allThreadNums: %d\n", allThreadNums);

    // 获取当前block是grid中的第几个block的寻找方案
    int blockId = blockIdx.x + (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y);
    printf("blockId: %d\n", blockId);

    // 获取当前thread是block中的第几个thread的寻找方案
    int threadId = threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y);
    printf("threadId: %d\n", threadId);

    // 获取当前thread是gird中的第几个thread的寻找方案
    int threadNumsInBlock = blockDim.x * blockDim.y * blockDim.z;
    int realThreadId = threadId + (blockId * threadNumsInBlock);
    printf("real threadId: %d\n", realThreadId);
}

void launchThreadBlockGridTest(int allThreadNum) {
    dim3 grid((allThreadNum + 1023)/1024); // 1023向上取整, 表示grid中包含ceil(allThreadNum)个grid
    dim3 block(1024); // 表示一个block中包含1024个thread
    threadBlockGridTest<<<grid, block>>>();
}

int main() {
    int allThreadNum = 2048;
    launchThreadBlockGridTest(allThreadNum);
    cudaDeviceSynchronize();
    return 0;
}