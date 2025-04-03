#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define THREAD_PER_BLOCK 256

__global__ void reduce0(float *d_input, float *d_output) {
    // 从block的角度去思考要如何写，首先把索引重新改到block的初始位置
    float *d_input_begin = d_input + blockIdx.x * blockDim.x;

    // 然后思考block中的每个线程要执行什么操作
    // 1, 2, 4
    for (int i = 1; i < blockDim.x; i = i * 2) {
        if (threadIdx.x % (i * 2) == 0) {
            d_input_begin[threadIdx.x] += d_input_begin[threadIdx.x + i];
        }
        // 同步
        __syncthreads();
    }

    // 把block的结果放到output中，只需要让第一个thread执行即可
    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = d_input_begin[0];
    }
}

bool check(float *output, float *result, int block_num) {
    for (int i = 0; i < block_num; i++) {
        if (abs(output[i] - result[i]) > 1e-2) {
            printf("in index %d, output: %lf vs. result: %lf\n", i, output[i], result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    float *input = (float*)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float)); // 传递了二级指针

    int block_num = N / THREAD_PER_BLOCK;
    float *output = (float*)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    float *result = (float*)malloc(block_num * sizeof(float));
    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    // cpu的reduce计算, 就是每个block都单独去并行计算, 把结果都放在result里面
    // result只包含`block_num`个元素，每个元素代表一个block的计算结果
    for (int i = 0; i < block_num; i++) {
        float cur = 0.0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++) {
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    // 把cpu mem上的数据拷贝到gpu mem上
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);  
    // grid数量只有1，block数量为block_num，每个block的thread数量为THREAD_PER_BLOCK
    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
      
    reduce0<<<Grid, Block>>>(d_input, d_output);
    // 把结果拷贝回来
    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num)) {
        printf("all right\n");
    } else {
        printf("some wrong\n");
    }
    
    // 释放资源
    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(result);
    
    return 0;
}