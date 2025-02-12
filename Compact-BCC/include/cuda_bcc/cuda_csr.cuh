#ifndef CUDA_CSR_H
#define CUDA_CSR_H

#include <cub/cub.cuh>
#include <cuda_runtime.h>

void create_duplicate(int* d_u, int* d_v, int* d_u_out, int* d_v_out, long size, cudaStream_t stream = 0);
void gpu_csr(cub::DoubleBuffer<int>& d_keys, cub::DoubleBuffer<int>& d_values, void *d_temp_storage, long num_items, const int numvert, long* d_vertices, cudaStream_t stream = 0);

#endif // CUDA_CSR_H