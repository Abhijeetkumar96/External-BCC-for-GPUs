//---------------------------------------------------------------------
// CUDA & CUB Libraries
//---------------------------------------------------------------------
#include <cub/cub.cuh>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// CUDA Utility & helper functions
//---------------------------------------------------------------------
#include "utility/timer.hpp"
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

extern int local_block_size;

#define DEBUG

__global__
void init_bcc_num_kernel(int* d_mapping, int* d_imp_bcc_num, int numVert) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numVert) {
        d_imp_bcc_num[idx] = idx;
        d_mapping[idx] = idx;
    }
}

void print_mem_info() {
    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte), "Error: cudaMemGetInfo fails");

    double free_db = static_cast<double>(free_byte); 
    double total_db = static_cast<double>(total_byte);
    double used_db = total_db - free_db;
    std::cout << "----------------------------------------\n"
          << "GPU Memory Usage Post-Allocation:\n"
          << "Used:     " << used_db / (1024.0 * 1024.0) << " MB\n"
          << "Free:     " << free_db / (1024.0 * 1024.0) << " MB\n"
          << "Total:    " << total_db / (1024.0 * 1024.0) << " MB\n"
          << "========================================\n\n";
}

GPU_BCG::GPU_BCG(int vertices, long num_edges, long _batchSize) : numVert(vertices), orig_numVert(vertices), orig_numEdges(num_edges), batchSize(_batchSize) {

    numEdges = (2 * batchSize) + (2 * numVert);
    max_allot = numEdges;
    E = numEdges * 2;
    
    Timer myTimer;

    /* ------------------------------ Batch Spanning Tree ds starts ------------------------------ */

    std::cout << "batchSize: " << batchSize << std::endl;
    
    size_t batch_alloc = batchSize * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_edgelist, batch_alloc),       "Allocation error");
    CUDA_CHECK(cudaMalloc(&d_parentEdge, sizeof(uint64_t) * numVert),       "Failed to allocate memory for d_parentEdge");
    CUDA_CHECK(cudaMalloc(&d_rep, sizeof(int) * numVert),                   "Failed to allocate memory for d_rep_hook");
    CUDA_CHECK(cudaMalloc((void **)&d_flag, sizeof(int)),                   "Failed to allocate memory for c_flag");
    /* ------------------------------ Spanning Tree ds ends ------------------------------ */

    /* ------------------------------ BCC ds starts ------------------------------ */
    CUDA_CHECK(cudaMalloc((void**)&updated_edgelist, numEdges * sizeof(uint64_t)), "Failed to allocate original_edgelist array");
    CUDA_CHECK(cudaMalloc((void**)&d_edge_buffer,    numEdges * sizeof(uint64_t)), "Failed to allocate d_edge_buffer array");

    CUDA_CHECK(cudaMalloc((void**)&d_mapping,        numVert * sizeof(int)), "Failed to allocate d_mapping array");
    CUDA_CHECK(cudaMalloc((void**)&d_imp_bcc_num,    numVert * sizeof(int)), "Failed to allocate d_imp_bcc_num array");

    // set these two arrays (one time work)
    long threadsPerBlock = maxThreadsPerBlock;
    size_t blocks = (numVert + threadsPerBlock - 1) / threadsPerBlock;

    // Becoz zeroth batch is a spanning tree and in a spanning every vertex is a cut vertex and ibcc
    // number of each vertex is the number itself
    init_bcc_num_kernel<<<blocks, threadsPerBlock>>>(d_mapping, d_imp_bcc_num, numVert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to set bcc numbers kernel");

    /* ------------------------------ step 1: Eulerian Tour ds starts ------------------------------ */

    int euler_edges = 2 * numVert - 2;

    CUDA_CHECK(cudaMalloc((void **)&d_first, sizeof(int) * numVert), "Failed to allocate d_first");
    CUDA_CHECK(cudaMalloc((void **)&d_last,  sizeof(int) * numVert),  "Failed to allocate d_last");

    CUDA_CHECK(cudaMalloc((void **)&d_edges_to,     sizeof(int) * euler_edges), "Failed to allocate d_edges_to");
    CUDA_CHECK(cudaMalloc((void **)&d_edges_from,   sizeof(int) * euler_edges), "Failed to allocate d_edges_from");
    CUDA_CHECK(cudaMalloc((void **)&d_next,         sizeof(int) * euler_edges), "Failed to allocate memory for d_next");
    CUDA_CHECK(cudaMalloc((void **)&succ,           sizeof(int) * euler_edges), "Failed to allocate succ array");
    CUDA_CHECK(cudaMalloc((void **)&devRank,        sizeof(int) * euler_edges), "Failed to allocate devRank array");
    CUDA_CHECK(cudaMalloc((void **)&devW1Sum,       sizeof(int) * euler_edges), "Failed to allocate devW1Sum");

    CUDA_CHECK(cudaMalloc((void **)&d_index, sizeof(uint64_t) * euler_edges), "Failed to allocate memory for d_index");

    CUDA_CHECK(cudaMalloc((void **)&d_roots, sizeof(int)), "Failed to allocate memory for d_roots");
    // List Ranking Params
    CUDA_CHECK(cudaMallocHost((void **)&notAllDone, sizeof(int)), "Failed to allocate notAllDone");
    CUDA_CHECK(cudaMalloc((void **)&devRankNext, sizeof(ull) * euler_edges), "Failed to allocate devRankNext");
    CUDA_CHECK(cudaMalloc((void **)&devNotAllDone, sizeof(int)), "Failed to allocate devNotAllDone");
    // List Ranking Params end

    // output arrays
    CUDA_CHECK(cudaMalloc((void **)&d_parent,       sizeof(int) * numVert), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc((void **)&d_level,        sizeof(int) * numVert),  "Failed to allocate d_level");
    CUDA_CHECK(cudaMalloc((void **)&d_first_occ,    sizeof(int) * numVert), "Failed to allocate memory for d_first_occ");
    CUDA_CHECK(cudaMalloc((void **)&d_last_occ,     sizeof(int) * numVert), "Failed to allocate memory for d_last_occ");
    /* ------------------------------ Eulerian Tour ds ends ------------------------------ */

    /* ------------------------------------ step 2: BCC ds starts --------------------------------------- */
    
    CUDA_CHECK(cudaMalloc(&d_left,  sizeof(int) * numVert), "Failed to allocate memory for d_left");
    CUDA_CHECK(cudaMalloc(&d_right, sizeof(int) * numVert), "Failed to allocate memory for d_right");

    int n_asize = (2*numVert + local_block_size - 1) / local_block_size;

    CUDA_CHECK(cudaMalloc(&d_na1, n_asize * sizeof(int)) , "Failed to allocate memory to d_na1");
    CUDA_CHECK(cudaMalloc(&d_na2, n_asize * sizeof(int)), "Failed to allocate memory to d_na1");
    

    // For updating bcc numbers
    CUDA_CHECK(cudaMalloc(&d_bcc_flag, sizeof(int) * numVert), "Failed to allocate memory for d_bcc_flag");
    CUDA_CHECK(cudaMalloc(&iscutVertex, sizeof(int) * numVert), "Failed to allocate memory for iscutVertex");

    // This is for removing self loops and duplicates
    CUDA_CHECK(cudaMalloc((void**)&d_flags, numEdges * sizeof(unsigned char)), "Failed to allocate flag array");
    CUDA_CHECK(cudaMallocHost(&h_num_selected_out, sizeof(long)),   "Failed to allocate pinned memory for h_num_items value");
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(long)), "Failed to allocate d_num_selected_out");

    // Repair data-structure
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int) * numVert),   "Failed to allocate memory for d_counter");
    /* ------------------------------------ BCC ds ends --------------------------------------- */

    // pinned memories
    CUDA_CHECK(cudaMallocHost(&h_max_ps_bcc, sizeof(int)),                    "Failed to allocate pinned memory for max_ps_bcc value");
    CUDA_CHECK(cudaMallocHost(&h_max_ps_cut_vertex, sizeof(int)),             "Failed to allocate pinned memory for max_ps_cut_vertex value");

    // Initialize GPU memory
    auto dur = myTimer.stop();

    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte), "Error: cudaMemGetInfo fails");

    double free_db = static_cast<double>(free_byte); 
    double total_db = static_cast<double>(total_byte);
    double used_db = total_db - free_db;

    std::cout << "========================================\n"
          << "       Allocation Details & GPU Memory Usage \n"
          << "========================================\n\n"
          << "Vertices (numVert):             " << numVert << "\n"
          << "Edges (numEdges):               " << numEdges << "\n"
          << "Max Allotment:                  " << max_allot << "\n"
          << "Device Allocation & Setup Time: " << dur << " ms\n"
          << "Batch Size:                     " << batchSize << "\n"
          << "Total Number of Batches:        " << (orig_numEdges + batchSize - 1) / batchSize << "\n" 
          << "----------------------------------------\n"
          << "GPU Memory Usage Post-Allocation:\n"
          << "Used:     " << used_db / (1024.0 * 1024.0) << " MB\n"
          << "Free:     " << free_db / (1024.0 * 1024.0) << " MB\n"
          << "Total:    " << total_db / (1024.0 * 1024.0) << " MB\n"
          << "========================================\n\n";
}