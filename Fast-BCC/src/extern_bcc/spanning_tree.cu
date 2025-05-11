#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "extern_bcc/bcg_memory_utils.cuh"
#include "extern_bcc/spanning_tree.cuh"
#include "utility/cuda_utility.cuh"

// #define DEBUG
#define num_threads 1024

__device__ 
inline int custom_compress(int i, int* temp_label) {
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
inline bool union_async(long idx, int src, int dst, int* temp_label, uint64_t* edges, uint64_t* st_edges) {
    while (1) {
        int u = custom_compress(src, temp_label);
        int v = custom_compress(dst, temp_label);
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
void init_parent_label(int* d_rep, uint64_t* d_parentEdge, int numVert){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numVert){
        d_rep[idx]  = idx;
        d_parentEdge[idx] = INT_MAX;
    }
}

__global__ 
void update_edgeslist_kernel(
    const uint64_t *d_edges_input, 
    uint64_t *d_edges_output, 
    const int root,
    int N) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thid < N) {

        if (thid == root)
          return;

        int afterRoot = thid > root;
        d_edges_output[thid - afterRoot] = d_edges_input[thid];;
    }
}

void update_edgelist(uint64_t* old_edgelist, uint64_t* updated_edgelist, int root, int nodes) {

    int blockSize = 1024;
    int numBlocks = ((nodes - 1) + blockSize - 1) / blockSize; 

    // Launch the kernel
    update_edgeslist_kernel<<<numBlocks, blockSize>>>(
        old_edgelist,
        updated_edgelist,
        root, 
        nodes);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_edgeslist kernel");
}

__global__ 
void union_find_gpu(long total_elt, int* temp_label, uint64_t* edges, uint64_t* st_edges) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elt) {
        int u = edges[idx] >> 32;
        int v = edges[idx] & 0xFFFFFFFF;
        if (u < v) {
            union_async(idx, u, v, temp_label, edges, st_edges);
        }
    }
}

void run_union_find(
    int numVert, long numEdges, 
    uint64_t* d_edgelist, uint64_t* d_parentEdge, int* temp_label) {
    
    long grid_size_union = (numEdges + num_threads - 1) / num_threads;

    union_find_gpu<<<grid_size_union, num_threads>>>( 
        numEdges, 
        temp_label, 
        d_edgelist, 
        d_parentEdge 
    );
}

int construct_batch_spanning_tree(GPU_BCG& g_bcg_ds) {

    int nodes = g_bcg_ds.numVert;
    long edges = g_bcg_ds.orig_numEdges;
    long batch_size = g_bcg_ds.batchSize;

    std::cout << "Nodes: " << nodes << ", edges: " << edges << " and batchSize: " << batch_size << "\n";

    const uint64_t* h_edgelist  =   g_bcg_ds.h_edgelist;
    uint64_t* d_edgelist        =   g_bcg_ds.d_edgelist;
    uint64_t* d_parentEdge      =   g_bcg_ds.d_parentEdge;     // output st edges
    
    int* d_rep                  =   g_bcg_ds.d_rep; // d_rep is temp_label

    int num_blocks_vert = (nodes + num_threads - 1) / num_threads;

    init_parent_label<<<num_blocks_vert, num_threads>>>(d_rep, d_parentEdge, nodes);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init kernel");

    auto start = std::chrono::high_resolution_clock::now();

    long num_batches = edges / batch_size;
    if (edges % batch_size != 0) {
        num_batches++;
    }

    for (long i = 0; i < num_batches; i++) {
        long start = i * batch_size;
        long end = std::min((i + 1) * batch_size, edges);
        long num_elements_in_batch = end - start;    

        CUDA_CHECK(cudaMemcpy(
            d_edgelist, 
            h_edgelist + start, 
            num_elements_in_batch * sizeof(uint64_t), 
            cudaMemcpyHostToDevice), 
        "Memcpy error");

        // Run the union find algorithm.................................
        run_union_find(
            nodes, 
            num_elements_in_batch, 
            d_edgelist, 
            d_parentEdge, 
            d_rep);
    }

    int root = 0;
    std::cout << "Root Value: " << root << std::endl;

    if(g_verbose) {
        std::cout << "Spanning Tree Edges:" << std::endl;
        print_device_edges(d_parentEdge, nodes);
    }

    update_edgelist(g_bcg_ds.d_parentEdge, g_bcg_ds.updated_edgelist, root, g_bcg_ds.numVert);

    // zeroth batch is ready now
    // update the edge count for the zeroth batch
    g_bcg_ds.numEdges = g_bcg_ds.numVert - 1;

    #ifdef DEBUG
        std::cout << "Spanning Tree Edges after removing extra edge: " << std::endl;
        print_device_edges(g_bcg_ds.updated_edgelist, g_bcg_ds.numVert - 1);
    #endif

    return root;
}