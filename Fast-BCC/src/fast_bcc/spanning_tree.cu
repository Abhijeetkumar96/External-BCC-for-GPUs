#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "fast_bcc/euler.cuh"
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

#define num_threads 1024
#define ins_batches 10

__device__ 
inline int find_compress_1(int i, int* d_componentParent)
{
    int j = i;

    if(d_componentParent[j] == j) 
        return j;

    // pointer jumping level by level
    do {
        j = d_componentParent[j];
    } while (d_componentParent[j] != j);

    int tmp;
    while((tmp = d_componentParent[i]) > j) {
        d_componentParent[i] = j;
        i = tmp;
    }
    return j;
}


__device__ 
inline bool union_async_1(long i, long idx, uint64_t d_edge, uint64_t* d_parentEdge, int* d_componentParent) {

        int src = d_edge >> 32;  // Extract higher 32 bits
        int dst = d_edge & 0xFFFFFFFF; // Extract lower 32 bits

    while(1) {
        int u = find_compress_1(src, d_componentParent);
        int v = find_compress_1(dst, d_componentParent);

        if(u == v) 
            break;

        if(v > u) { 
            int temp; 
            temp = u; 
            u = v; 
            v = temp; 
        }

        // check this once for syntax
        if(u == atomicCAS(&d_componentParent[u], u, v)) {
            d_parentEdge[u] = d_edge;
            return true;
        } 
    }
    return false;
}


__global__ 
void union_find_gpu_COO_1(long batch_size, uint64_t* d_edges, uint64_t* d_parentEdge, int* d_componentParent , long offset){
    long idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if(idx < batch_size + offset) {
        bool r = union_async_1(idx, idx, d_edges[idx], d_parentEdge, d_componentParent);
    }
}

__global__ 
void cc_gpu_1(int numVert, int* d_rep, int* d_componentParent) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numVert) {
        d_rep[idx] = find_compress_1(idx, d_componentParent);
    }
}

__global__
void init_parent_label_1(int* d_componentParent, int* d_rep, uint64_t* d_parentEdge, int numVert){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numVert){
        d_componentParent[idx] = idx;
        d_rep[idx]  = idx;
        d_parentEdge[idx] = INT_MAX;
    }
}

void run_union_find_1(
    int numVert, long batch_size, 
    uint64_t* d_edges, uint64_t* d_parentEdge, 
    int* d_componentParent, int* d_rep){
    
    // long grid_size_union = (batch_size + num_threads - 1) / num_threads;
    // int grid_size_final = (numVert + num_threads - 1) / num_threads;

    // union_find_gpu_COO_1<<<grid_size_union, num_threads>>>(
    //     batch_size, d_edges, 
    //     d_parentEdge, d_componentParent);
    // cc_gpu_1<<<grid_size_final, num_threads>>>(numVert, d_rep, d_componentParent);
    //    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize cc_gpu");


	// long l1 = batch_size / ins_batches;
	// long l2 = l1;
	// long l3 = batch_size - l1 - l2;
	// long offset1 = 0;
	// long offset2 = l1;
	// long offset3 = l1 + l2;

	// long grid_size_union1 = (l1 + num_threads - 1) / num_threads;
	// long grid_size_union2 = (l2 + num_threads - 1) / num_threads;
	// long grid_size_union3 = (l3 + num_threads - 1) / num_threads;
	long grid_num = (numVert + num_threads - 1) / num_threads;

	// union_find_gpu_COO_1<<<grid_size_union1, num_threads>>>(
	// 	l1, d_edges, d_parentEdge, d_componentParent, offset1);
	// union_find_gpu_COO_1<<<grid_size_union2, num_threads>>>(
	// 	l2, d_edges, d_parentEdge, d_componentParent, offset2);
	// union_find_gpu_COO_1<<<grid_size_union3, num_threads>>>(
	// 	l3, d_edges, d_parentEdge, d_componentParent, offset3);

    for(int i=0;i<ins_batches;i++){
        long l = batch_size / ins_batches;
        long offset = i*l;
        if(i == ins_batches-1){
            l = batch_size - offset;
        }
        long grid_size_union = (l + num_threads - 1) / num_threads;
        union_find_gpu_COO_1<<<grid_size_union, num_threads>>>(
            l, d_edges, d_parentEdge, d_componentParent, offset);
    }
	cc_gpu_1<<<grid_num, num_threads>>>(numVert, d_rep, d_componentParent);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize cc_gpu");
}



int construct_spanning_tree(GPU_BCG& g_bcg_ds) {

    uint64_t* d_edgelist 		= 	g_bcg_ds.updated_edgelist; // The input edgelist
    int nodes 					= 	g_bcg_ds.numVert;
    long edges 					= 	g_bcg_ds.numEdges;
    uint64_t* d_parentEdge 		= 	g_bcg_ds.d_parentEdge; // The actual spanning Tree edges (output)
    int* d_componentParent 		= 	g_bcg_ds.d_componentParent;
    int* d_rep 					= 	g_bcg_ds.d_rep;

    #ifdef DEBUG
        std::cout << "Actual Edges array:\n";
        print_device_edges(d_edgelist, edges);
    #endif

    int num_blocks_vert = (nodes + num_threads - 1) / num_threads;

    init_parent_label_1<<<num_blocks_vert, num_threads>>>(d_componentParent, d_rep, d_parentEdge, nodes);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init kernel");

    auto start = std::chrono::high_resolution_clock::now();

    run_union_find_1(
        nodes, 
        edges, 
        d_edgelist, 
        d_parentEdge, 
        d_componentParent, 
        d_rep);
    
    int root = 0;
    // std::cout << "Root Value: " << root << std::endl;

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
