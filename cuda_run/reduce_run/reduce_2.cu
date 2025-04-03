// 本代码将使用减少wrap内的判断分化来改造reduce_1.cu，目的是提高GPU执行的效率
// 核心就是让GPU的一个wrap内的线程的逻辑保持尽可能的一致，而不是让wrap内线程分别处理不同的逻辑

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce_2(float *d_input, float *d_output) {
    // 首先, 需要让block中的每个元素都拷贝到shared memory上
    float *d_input_begin = d_input + blockDim.x * blockIdx.x;
    __shared__ float shared_d_input[THREAD_PER_BLOCK]; // 需要单独申请一段shared memory空间
    for (int i = 0; i < blockDim.x; i++) {
        shared_d_input[i] = d_input_begin[i];
    }
    __syncthreads(); // 因为拷贝到shared memory也是并行化操作，所以需要同步thread
    
    for (int i = 1; i < blockDim.x; i = i * 2) {
        // 这个操作就是让前面一半的thread都进入if逻辑，后面一半的thread不进入if逻辑
        if (threadIdx.x * (2 * i) < blockDim.x) {
            // 进入if逻辑后，每个thread其实不一定要处理对应位置的内存，因为毕竟还是shared memory
            // 因此在第一层： thread0 处理的是 0+1 => 放到0, thread1 处理的是 2+3 => 放到2, thread2 处理的是 4+5 => 放到4, thread3 处理的是 6+7 => 放到6
            // 在第二层: thread0处理的是 0+2 => 放到0, thread1 处理的是 4+6 => 放到4
            int index = threadIdx.x * (2 * i);
            shared_d_input[index] += shared_d_input[index + i];
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = shared_d_input[0];
    }
}

bool check(float *output, float *result, int block_num) {
    for (int i = 0; i < block_num; i++) {
        if (abs(output[i] - result[i]) > 1e-2) {
            printf("index %d wrong\n", i);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 2 * 1024 * 1024;
    int block_num = N / THREAD_PER_BLOCK;
    float *input = (float*)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    float *output = (float*)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    

    float *result = (float*)malloc(block_num * sizeof(float));
    for (int i = 0; i < block_num; i++) {
        float res = 0.0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++) {
            res += input[i * THREAD_PER_BLOCK + j];
        }
        result[i] = res;
    }

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    reduce_2<<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num)) {
        printf("all ok\n");
    } else {
        printf("some wrong\n");
    }

    free(result);
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}