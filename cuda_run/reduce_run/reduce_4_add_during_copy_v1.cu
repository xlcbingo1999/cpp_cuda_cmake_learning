// 本代码将在数据从global memory拷贝到shared memory时，同时进行一次add操作，让效率更高一些

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce_2(float *d_input, float *d_output) {
    // 首先, 需要让block中的每个元素都拷贝到shared memory上
    float *d_input_begin = d_input + blockDim.x * blockIdx.x * 2; // 现在一次读取都是跨两个block的
    __shared__ float shared_d_input[THREAD_PER_BLOCK]; // 需要单独申请一段shared memory空间，大小也不需要变化
    shared_d_input[threadIdx.x] = d_input_begin[threadIdx.x] + d_input_begin[threadIdx.x + blockDim.x];
    __syncthreads(); // 因为拷贝到shared memory也是并行化操作，所以需要同步thread
    
    // 避免让两个thread同时处理一个wrap中的bank，因此每次都是跨很大的距离进行求和
    for (int i = blockDim.x / 2; i > 0; i = i / 2) {
        // 这个操作就是让前面一半的thread都进入if逻辑，后面一半的thread不进入if逻辑
        if (threadIdx.x < i) {
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
    int block_num = N / THREAD_PER_BLOCK / 2; // 只需要一半的数据被block覆盖即可
    float *input = (float*)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    float *output = (float*)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    

    float *result = (float*)malloc(block_num * sizeof(float));
    for (int i = 0; i < block_num; i++) {
        float res = 0.0;
        for (int j = 0; j < THREAD_PER_BLOCK * 2; j++) {
            res += input[i * THREAD_PER_BLOCK * 2 + j];
        }
        result[i] = res;
    }

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1); // 一个block中的thread数量还是不需要变的，只需要让一半的数据被block覆盖即可

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