#include <iostream>
#include <cuda_runtime.h>

#include "utility/cuda_utility.cuh"
#include "extern_bcc/extern_bcc.cuh"       
#include "extern_bcc/bcg_memory_utils.cuh"


#define tb_size 1024

__device__ inline 
int find_compress(int i, int* temp_label) {
    int j = i;
    if (temp_label[j] == j) {
        return j;
    }
    do {
        j = temp_label[j];
    } while (temp_label[j] != j);

    int tmp;
    while((tmp = temp_label[i])>j) {
        temp_label[i] = j;
        i = tmp;
    }
    return j;
}


__device__ inline 
bool union_async(long idx, int src, int dst, int* temp_label) {
    // printf("Doing union_async for src: %d and dst: %d\n", src, dst);
    while(1) {
        int u = find_compress(src, temp_label);
        int v = find_compress(dst, temp_label);

        if(u == v) break;
        if(v > u) { int temp; temp = u; u = v; v = temp; }
        if(u == atomicCAS(&temp_label[u], u, v)) {
           return true;
        } 
    }
    return false;
}

__global__ 
void union_find_gpu_COO(
    int numVert, int* temp_label, 
    int* d_parent, uint64_t* d_sf_edges,
    int* d_first, int* d_last, 
    int* d_low, int* d_high) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        // The first group of n threads operates on d_parent.
        int u = idx;
        int v = d_parent[idx];

        if(u == v) return;

        int f_u = d_first[u];
        int l_u = d_last[u];

        int f_v = d_first[v];
        int l_v = d_last[v];
        
        int low_u = d_low[u];
        int high_u = d_high[u];
        
        int low_v = d_low[v];
        int high_v = d_high[v];
        
        // if it a fence edge; ignore, else do CC.
        if( (f_u <= low_v and l_u >= high_v) or (f_v <= low_u and l_v >= high_u) ) return;

        // printf("Calling union_async for tree edges src: %d and dst: %d\n", u, v);
        union_async(idx, u, v, temp_label);
        
    } else if (idx < 2 * numVert) {
        // The second group of n threads operates on d_sf_edges.
        int index = idx - numVert;
        if(d_sf_edges[index] == INT_MAX) {
            // printf("Returning for the cross-edges src: %ld\n", d_sf_edges[index]);    
            return;
        }
        // all these are cross-edges; so do CC for all
        int u = (d_sf_edges[index] >> 32) & 0xFFFFFFFF;
        int v = (d_sf_edges[index]) & 0xFFFFFFFF;
        // #ifdef DEBUG
        // printf("Calling union_async for cross-edges src: %d and dst: %d\n", u, v);
        // #endif
        union_async(idx, u, v, temp_label);
    }
}

__global__ 
void union_find_gpu_repair(
    int numVert, int* temp_label, int* d_counter,
    int* d_parent, uint64_t* d_sf_edges,
    int* d_first, int* d_last, 
    int* d_low, int* d_high) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        // The first group of n threads operates on d_parent.
        int u = idx;
        int v = d_parent[idx];

        if(u == v) return;

        int f_v = d_first[v];
        int l_v = d_last[v];
        
        int low_u = d_low[u];
        int high_u = d_high[u];
        
        // if it a fence edge; ignore, else do CC.
        if(f_v <= low_u and l_v >= high_u and d_counter[v] == 1) {
            // printf("u:%d, v:%d is a fence edge; counter[u]: %d and counter[v]:%d.\n", u, v, d_counter[u], d_counter[v]);
                return;
        }

        // printf("Calling union_async for tree edges src: %d and dst: %d\n", u, v);
        union_async(idx, u, v, temp_label);
        
    } else if (idx < 2 * numVert) {
        // The second group of n threads operates on d_sf_edges.
        int index = idx - numVert;
        if(d_sf_edges[index] == INT_MAX) {
            // printf("Returning for the cross-edges src: %ld\n", d_sf_edges[index]);    
            return;
        }

        // all these are cross-edges; so do CC for all
        int u = (d_sf_edges[index] >> 32) & 0xFFFFFFFF;
        int v = (d_sf_edges[index]) & 0xFFFFFFFF;
        // #ifdef DEBUG
        // printf("Calling union_async for cross-edges src: %d and dst: %d\n", u, v);
        // #endif
        union_async(idx, u, v, temp_label);
    }
}

__global__ 
void cc_gpu(int* label, int* temp_label, int V) {       
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < V) {
        label[idx] = find_compress(idx, temp_label);
    }
}

__global__
void init_parent_label(int* temp_label, int* label, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < V) {
        temp_label[idx] = idx;
        label[idx] = idx;
    }
}


void CC(GPU_BCG& g_bcg_ds, const bool lastBatch) {

    uint64_t* d_sf_edges = g_bcg_ds.d_index;
    int V                = g_bcg_ds.numVert;

    int* d_parent           = g_bcg_ds.d_parent;

    int* d_first            = g_bcg_ds.d_first_occ;
    int* d_last             = g_bcg_ds.d_last_occ;

    int* d_low              = g_bcg_ds.d_low;
    int* d_high             = g_bcg_ds.d_high;

    int* temp_label         = g_bcg_ds.d_fg;
    int* label              = g_bcg_ds.d_rep;

    int* d_counter          = g_bcg_ds.d_counter;

    int grid_size_final = CEIL(V, tb_size);

    init_parent_label<<<grid_size_final, tb_size>>>(
        temp_label, 
        label, 
        V);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init_parent_label_cc kernel");

    grid_size_final = CEIL(2*V, tb_size);
    if(!lastBatch) {
        union_find_gpu_COO<<<grid_size_final, tb_size>>>(
            V, temp_label, 
            d_parent, d_sf_edges,
            d_first, d_last,
            d_low, d_high);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize union_find_gpu_COO_cc kernel");
    }
    else {
        union_find_gpu_repair<<<grid_size_final, tb_size>>>(
            V, temp_label, d_counter, 
            d_parent, d_sf_edges,
            d_first, d_last,
            d_low, d_high);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize union_find_gpu_COO_cc kernel");        
    }

    grid_size_final = CEIL(V, tb_size);
    cc_gpu<<<grid_size_final, tb_size>>>(label, temp_label, V);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize cc_gpu_cc kernel");

    #ifdef DEBUG
        std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        std::cout << "BCC Numbers from CC function:" << "\n";
        std::cout << "V from CC: " << V << std::endl;
        print_device_array(label, V);
        std::cout << "\nPrinting from CC complete" << std::endl;
        std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" << std::endl;
    #endif
}
