#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

#include <cuda_runtime.h>

// #define DEBUG

__global__ 
void init_kernel(int *d_counter, int numVerts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVerts) {
        d_counter[tid] = 0;
    }
}

__global__ 
void freq_cal(int *d_mapping, int *d_counter, int orig_numVerts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < orig_numVerts) {
        int ver = d_mapping[tid];
        if (d_counter[ver] < 2) {
            atomicAdd(d_counter + ver, 1);
        }
    }
}

void repair(GPU_BCG& g_bcg_ds) {
    int numVerts = g_bcg_ds.numVert;
    int orig_numVerts = g_bcg_ds.orig_numVert;
    
    int *d_counter = g_bcg_ds.d_counter;
    int *d_mapping = g_bcg_ds.d_mapping;
    
    int threads = 1024;
    int numBlocks = (numVerts + threads - 1) / threads;
    
    init_kernel<<<numBlocks, threads>>>(d_counter, numVerts);
    CUDA_CHECK(cudaDeviceSynchronize(), "init_kernel synchronization failed");
    
    numBlocks = (orig_numVerts + threads - 1) / threads;
    freq_cal<<<numBlocks, threads>>>(d_mapping, d_counter, orig_numVerts);
    CUDA_CHECK(cudaDeviceSynchronize(), "freq_cal synchronization failed");
    
    #ifdef DEBUG
        std::cout << "Printing from Repair function" << std::endl;
        std::cout << "d_counter array:" << std::endl;
        DisplayDeviceArray(d_counter, numVerts);
    #endif
}
