#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
// #include <device_functions.h>

void initData(float* data, size_t nElem) {
    for (size_t i = 0; i < nElem; i++) {
        data[i] = i % 255;
    }
}

void sumArraysOnHost(float* hA, float* hB, float* hostRef, size_t nElem) {
    for (size_t i = 0; i < nElem; i++) {
        hostRef[i] = hA[i] + hB[i];
    }
}

__global__ void sumArrayOnGPU(float* dA, float* dB, float* deviceRef, size_t nElem) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < nElem) {
        deviceRef[tid] = dA[tid] + dB[tid];
    }
}

bool checkResults(float* hostRef, float* deviceRef, size_t nElem) {
    for (size_t i = 0; i < nElem; i++) {
        if (abs(hostRef[i] - deviceRef[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

int main() {
    // 设置执行的设备
    int nDev = 0;
    cudaSetDevice(nDev);

    // 获取特定设备的信息
    cudaDeviceProp stDeviceProp;
    cudaGetDeviceProperties(&stDeviceProp, nDev);
    
    if (!stDeviceProp.canMapHostMemory) {
        printf("Device %d does not support CPU host mem\n", nDev);
        cudaDeviceReset();
    
        return 0;
    }

    printf("Using Device %d: %s\n", nDev, stDeviceProp.name);


    int nPower = 10;
    int nElem = 1 << nPower;
    size_t nBytes = nElem * sizeof(float);

    float *hA, *hB, *hostRef, *deviceRef;
    hA = (float*)malloc(nBytes);
    hB = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    deviceRef = (float*)malloc(nBytes);

    initData(hA, nElem);
    initData(hB, nElem);
    memset(hostRef, 0, nBytes);
    memset(deviceRef, 0, nBytes);
    
    // 第一种: 直接在host上进行计算
    sumArraysOnHost(hA, hB, hostRef, nElem);

    // 第二种: 直接在device上进行计算
    float *dA, *dB, *dC;
    cudaMalloc(&dA, nBytes);
    cudaMalloc(&dB, nBytes);
    cudaMalloc(&dC, nBytes);
    cudaMemcpy(dA, hA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, nBytes, cudaMemcpyHostToDevice);
    int nThreads = 512;
    dim3 block(nThreads);
    dim3 grid((nElem + block.x - 1) / block.x);
    sumArrayOnGPU<<<grid, block>>>(dA, dB, dC, nElem);
    cudaMemcpy(deviceRef, dC, nBytes, cudaMemcpyDeviceToHost);
    bool success1 = checkResults(hostRef, deviceRef, nElem);
    printf("success1: %d\n", success1);
    cudaFree(dA);
    cudaFree(dB);
    free(hA);
    free(hB);

    // 第三种, 在host上申请zero-copy memory用于计算
    unsigned int nFlags = cudaHostAllocMapped;
    cudaHostAlloc(&hA, nBytes, nFlags);
    cudaHostAlloc(&hB, nBytes, nFlags);
    
    initData(hA, nElem);
    initData(hB, nElem);
    memset(hostRef, 0, nBytes);
    memset(deviceRef, 0, nBytes);

    cudaHostGetDevicePointer(&dA, hA, 0);
    cudaHostGetDevicePointer(&dB, hB, 0);
    
    sumArraysOnHost(hA, hB, hostRef, nElem);
    sumArrayOnGPU<<<grid, block>>>(dA, dB, dC, nElem);
    cudaMemcpy(deviceRef, dC, nBytes, cudaMemcpyDeviceToHost);
    bool success2 = checkResults(hostRef, deviceRef, nElem);
    printf("success2: %d\n", success2);
    cudaFree(dA);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    free(hostRef);
    free(deviceRef);

    cudaDeviceReset();
    
    return 0;
}