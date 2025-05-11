#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "fast_bcc/euler.cuh"
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

#define num_threads 1024

__device__ 
inline int custom_compress_(int i, int* temp_label) {
    int j = i;
    if (temp_label[j] == j) {
        return j;
    }
    do {
        j = temp_label[j];
    } while (temp_label[j] != j);

    int tmp;
    while ((tmp = temp_label[i]) > j) {
        temp_label[i] = j;
        i = tmp;
    }
    return j;
}

__device__ 
inline bool union_async_(long idx, int src, int dst, int* temp_label, uint64_t* edges, uint64_t* st_edges) {
    while (1) {
        int u = custom_compress_(src, temp_label);
        int v = custom_compress_(dst, temp_label);
        if (u == v) break;
        if (v > u) { int temp; temp = u; u = v; v = temp; }
        if (u == atomicCAS(&temp_label[u], u, v)) {
           st_edges[u] = edges[idx];
           return true;
        }
    }
    return false;
}

__global__
void init_parent_label_(int* d_rep, uint64_t* d_parentEdge, int numVert){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numVert){
        d_rep[idx]  = idx;
        d_parentEdge[idx] = INT_MAX;
    }
}

__global__ 
void union_find_st(long total_elt, int* temp_label, uint64_t* edges, uint64_t* st_edges) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elt) {
        int u = edges[idx] >> 32;
        int v = edges[idx] & 0xFFFFFFFF;
        if (u < v) {
            union_async_(idx, u, v, temp_label, edges, st_edges);
        }
    }
}

void run_union_find_(
    int numVert, long numEdges, 
    uint64_t* d_edgelist, uint64_t* d_parentEdge, int* temp_label) {
    
    long grid_size_union = (numEdges + num_threads - 1) / num_threads;

    union_find_st<<<grid_size_union, num_threads>>>( 
        numEdges, 
        temp_label, 
        d_edgelist, 
        d_parentEdge 
    );
}

int construct_spanning_tree(GPU_BCG& g_bcg_ds) {

    uint64_t* d_edgelist        =   g_bcg_ds.updated_edgelist; // The input edgelist
    int nodes                   =   g_bcg_ds.numVert;
    long edges                  =   g_bcg_ds.numEdges;
    uint64_t* d_parentEdge      =   g_bcg_ds.d_parentEdge; // The actual spanning Tree edges (output)
    int* d_rep                  =   g_bcg_ds.d_rep;

    #ifdef DEBUG
        std::cout << "Actual Edges array:\n";
        print_device_edges(d_edgelist, edges);
    #endif

    int num_blocks_vert = (nodes + num_threads - 1) / num_threads;

    init_parent_label_<<<num_blocks_vert, num_threads>>>(d_rep, d_parentEdge, nodes);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init kernel");

    auto start = std::chrono::high_resolution_clock::now();

    run_union_find_(
            nodes, 
            edges, 
            d_edgelist, 
            d_parentEdge, 
            d_rep);
    
    int root = 0;

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "One round of spanning tree construction takes: " << dur << " ms." << "\n";

    start = std::chrono::high_resolution_clock::now();
    cuda_euler_tour(nodes, root, g_bcg_ds);

    // end = std::chrono::high_resolution_clock::now();
    // dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "One round of Eulerian Tour takes: " << dur << " ms." << "\n";

    return root;
}
