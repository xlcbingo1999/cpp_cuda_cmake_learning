// 本代码将使用shared_memory改造reduce_0.cu，目的是为了减少全局内存的访问次数
// - Global Memory: GPU和CPU之间共享的一块大内存
// - Shared Memory: GPU层级架构中，每个block都有一块小内存，用于存储block内的线程共享的数据
//                  共享内存的访问速度比全局内存快，但是共享内存的大小有限制，一般不超过64KB、


#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce_1(float *d_input, float *d_output) {
    // 首先, 需要让block中的每个元素都拷贝到shared memory上
    float *d_input_begin = d_input + blockDim.x * blockIdx.x;
    __shared__ float shared_d_input[THREAD_PER_BLOCK]; // 需要单独申请一段shared memory空间
    for (int i = 0; i < blockDim.x; i++) {
        shared_d_input[i] = d_input_begin[i];
    }
    __syncthreads(); // 因为拷贝到shared memory也是并行化操作，所以需要同步thread
    
    for (int i = 1; i < blockDim.x; i = i * 2) {
        if (threadIdx.x % (i * 2) == 0) {
            shared_d_input[threadIdx.x] += shared_d_input[threadIdx.x + i];
        }
        __syncthreads();
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
    reduce_1<<<Grid, Block>>>(d_input, d_output);

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