//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <omp.h>
#include <vector>
#include <cassert>
#include <numeric>
#include <iterator>
#include <iostream>
#include <functional>

//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "utility/timer.hpp"             
#include "utility/utility.hpp"           
#include "utility/cuda_utility.cuh"      

//---------------------------------------------------------------------
// Spanning Tree and Biconnected Components (BCC) related functions
//---------------------------------------------------------------------
#include "extern_bcc/bcg_memory_utils.cuh"
#include "extern_bcc/spanning_tree.cuh"
#include "extern_bcc/edge_cleanup.cuh"
#include "extern_bcc/extern_bcc.cuh"
#include "fast_bcc/fast_bcc.cuh"
#include "fast_bcc/repair.cuh"

// #define DEBUG

 /*
  ____                                ____        _       _     
 |  _ \ _ __ ___   ___ ___  ___ ___  | __ )  __ _| |_ ___| |__  
 | |_) | '__/ _ \ / __/ _ \/ __/ __| |  _ \ / _` | __/ __| '_ \ 
 |  __/| | | (_) | (_|  __/\__ \__ \ | |_) | (_| | || (__| | | |
 |_|   |_|  \___/ \___\___||___/___/ |____/ \__,_|\__\___|_| |_|
                                                                
*/
__global__ 
void update_edges_kernel(
    int* bcc_num, int* d_mapping,                           // Mapping info
    uint64_t* d_prev_batch_edges,                           // Previous Graph edges
    uint64_t* d_new_batch_edges,                            // Newly copied edges
    uint64_t* d_edge_buffer,                               // Output arrays
    long prev_num_edges, long totalSize) {                  // Sizes for each array

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= totalSize) return;

    // if(idx == 0) {
    //     printf("BCC Number array:\n");
    //     for(int i = 0; i < 21; ++i) {
    //         printf("bcc[%d]: %d\n", i, bcc_num[i]);
    //     }
    // }
    
    if(idx < prev_num_edges) {

        uint64_t i = d_prev_batch_edges[idx];

        int u = i >> 32;  // Extract higher 32 bits
        int v = i & 0xFFFFFFFF; // Extract lower 32 bits

        #ifdef DEBUG
            // printf("Printing from update_edges_kernel:\n");
            printf("u: %d, v: %d, idx: %llu\n", u, v, idx);
        #endif
        
        int new_u = min(bcc_num[u], bcc_num[v]);
        int new_v = max(bcc_num[u], bcc_num[v]);

        d_edge_buffer[idx] = (static_cast<uint64_t>(new_u) << 32) | (new_v & 0xFFFFFFFFLL);

    } else {
        uint64_t i = d_new_batch_edges[idx - prev_num_edges];

        int u = i >> 32;  // Extract higher 32 bits
        int v = i & 0xFFFFFFFF; // Extract lower 32 bits
        
        #ifdef DEBUG
            printf("u: %d, v: %d, idx: %llu\n", u, v, idx);
        #endif

        int mapped_u = d_mapping[u];
        int mapped_v = d_mapping[v];

        int new_u = min(bcc_num[mapped_u], bcc_num[mapped_v]);
        int new_v = max(bcc_num[mapped_u], bcc_num[mapped_v]);

        d_edge_buffer[idx] = (static_cast<uint64_t>(new_u) << 32) | (new_v & 0xFFFFFFFFLL);
    }
}

/*
  _   _           _       _         __  __                   _             
 | | | |_ __   __| | __ _| |_ ___  |  \/  | __ _ _ __  _ __ (_)_ __   __ _ 
 | | | | '_ \ / _` |/ _` | __/ _ \ | |\/| |/ _` | '_ \| '_ \| | '_ \ / _` |
 | |_| | |_) | (_| | (_| | ||  __/ | |  | | (_| | |_) | |_) | | | | | (_| |
  \___/| .__/ \__,_|\__,_|\__\___| |_|  |_|\__,_| .__/| .__/|_|_| |_|\__, |
       |_|                                      |_|   |_|            |___/ 
*/

__global__
void update_mapping(int* d_mapping, int* bcc_num, int original_num_verts) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < original_num_verts) {
        d_mapping[idx] = bcc_num[d_mapping[idx]];
    }
}

void construct_BCG(GPU_BCG& g_bcg_ds, const size_t BatchSize, const bool isLastBatch) {

    int numVert                     =   g_bcg_ds.numVert;
    int original_num_verts          =   g_bcg_ds.orig_numVert;
    long numEdges                   =   g_bcg_ds.numEdges;
    long batchSize                  =   g_bcg_ds.batchSize;  

    uint64_t* d_prev_batch_edges    =   g_bcg_ds.updated_edgelist;
    uint64_t* d_new_batch_edges     =   g_bcg_ds.d_edgelist;

    // old_number_of_edges + new_batch_size
    long totalSize = g_bcg_ds.numEdges + BatchSize;
    // std::cout <<"Edge Count after adding new batch / totalSize: " << totalSize << "\n";
    assert(totalSize <= g_bcg_ds.max_allot);
    
    long threadsPerBlock = maxThreadsPerBlock;

    size_t blocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();    
    update_edges_kernel<<<blocks, threadsPerBlock>>>(
        g_bcg_ds.d_rep,                  // BCC numbers (int)
        g_bcg_ds.d_mapping,             // Vertex ID mapping (int)
        d_prev_batch_edges,             // Previous batch edges (uint64_t*)
        d_new_batch_edges,              // New batch of edges (uint64_t*)
        g_bcg_ds.d_edge_buffer,         // Output after remapping for edges (uint64_t*)
        g_bcg_ds.numEdges,              // Previous PBCG graph edge count (long)
        totalSize                       // Previous Edge Count + New Edge Count (long)
    );
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_edges_kernel");

    // std::cout << "Update edges kernel successfully completed." << "\n";
    blocks = (original_num_verts + threadsPerBlock - 1) / threadsPerBlock;
    
    // Update the actual vertex mapping array for the original graph
    update_mapping<<<blocks, threadsPerBlock>>>(
        g_bcg_ds.d_mapping,      // Vertex ID mapping
        g_bcg_ds.d_rep,  // Important BCC numbers
        original_num_verts       // Original number of vertices
    );
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_mapping");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // std::cout << "One round of transformation and mapping update takes: " << dur << " ms." << "\n";

    #ifdef DEBUG
        std::cout << "Updated Edgelist after transformation: " << "\n";
        print_device_edges(g_bcg_ds.d_edge_buffer, totalSize);
    #endif

    // std::cout << "Update mapping kernel successfully completed." << "\n";
   start = std::chrono::high_resolution_clock::now();
    // do clean-up, i.e. remove self loops and duplicates.
    remove_self_loops_duplicates(
        g_bcg_ds.d_edge_buffer,       // Input edge-stream 
        totalSize,                    // Total number of items
        g_bcg_ds.d_flags,             // Flags for marking
        g_bcg_ds.h_num_selected_out,  // Output: selected count
        g_bcg_ds.d_num_selected_out,  // Output: selected count
        g_bcg_ds.updated_edgelist    // Output: Updated Edgelist
    );

    g_bcg_ds.numEdges = totalSize;
    
    // Check if at anytime the alloted numEdges is becoming less than the current numEdges
    assert(g_bcg_ds.numEdges <= g_bcg_ds.max_allot); 

    // end = std::chrono::high_resolution_clock::now();
    // dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "One round of remove self loops takes: " << dur << " ms." << "\n";

    if(isLastBatch)
        repair(g_bcg_ds);

    // call cuda_bcc
    // std::cout << "Calling cuda_bcc\n";
        
    start = std::chrono::high_resolution_clock::now();
    Fast_BCC(g_bcg_ds, isLastBatch);

    end = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // std::cout << "\n*********** One round of BCC takes: " << dur << " ms. ***********n\n" << std::endl;
}

__global__
void init_bcc_num(int* parent, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n) {
        parent[tid] = tid;
    }
}

void extern_bcc(GPU_BCG& g_bcg_ds) {
    std::cout << "Started Spanning Tree construction." << std::endl;
    Timer myTimer;
    construct_batch_spanning_tree(g_bcg_ds);

    auto dur = myTimer.stop();
    std::cout <<"\nSpanning Tree finished in " << dur <<" ms.\n\n" << std::endl;

    // -------------------- Assigning to local variables -------------------- //
    const uint64_t* h_edgelist  =   g_bcg_ds.h_edgelist;
    uint64_t* d_edgelist        =   g_bcg_ds.d_edgelist;

    int nodes = g_bcg_ds.numVert;
    long edges = g_bcg_ds.orig_numEdges;
    long batch_size = g_bcg_ds.batchSize;

    int original_num_verts = g_bcg_ds.orig_numVert;

    // -------------------- *************************** -------------------- //

    int num_batches = edges / batch_size;
    if (edges % batch_size != 0) {
        num_batches++;
    }

    const long numThreads = 1024;
    int numBlocks = (nodes + numThreads - 1) / numThreads;

    init_bcc_num<<<numBlocks, numThreads>>>(g_bcg_ds.d_rep, nodes);
    CUDA_CHECK(cudaDeviceSynchronize(), "Error in launching initialise kernel");

    myTimer.reset();

    auto total = 0;

    for (int i = 0; i < num_batches; i++) {
        auto copy_time_start = std::chrono::high_resolution_clock::now();
        
        long start = i * batch_size;
        long end = std::min((i + 1) * batch_size, edges);
        long num_elements_in_batch = end - start;    

        CUDA_CHECK(cudaMemcpy(
            d_edgelist, 
            h_edgelist + start, 
            num_elements_in_batch * sizeof(uint64_t), 
            cudaMemcpyHostToDevice), 
        "Memcpy error");

        if(g_verbose) {
            std::cout << "\n\n\t******************************************\n";
            std::cout << "\t***          Batch " << i << ": " << start << " to " << end << "          ***\n";
            std::cout << "\tNumber of elements in batch: " << num_elements_in_batch << "\n";
            std::cout << "\t******************************************\n\n" << std::endl;
        }

        auto copy_time_end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(copy_time_end - copy_time_start).count();

        total += dur;

        bool isLastBatch = (i == num_batches - 1);

        construct_BCG(g_bcg_ds, num_elements_in_batch, isLastBatch);
    }

    std::cout << "Total Copy Times: " << total << " ms." << "\n" << std::endl;

    // long threadsPerBlock = maxThreadsPerBlock;
    // int blocks = (original_num_verts + threadsPerBlock - 1) / threadsPerBlock;

    // // Update the actual vertex mapping array for the original graph
    // update_mapping<<<blocks, threadsPerBlock>>>(
    //     g_bcg_ds.d_mapping,      // Vertex ID mapping
    //     g_bcg_ds.d_rep,  // Important BCC numbers
    //     original_num_verts       // Original number of vertices
    // );
    // CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_mapping");

    #ifdef DEBUG
        std::cout << "Mapping Array:" << "\n";
        print_device_array(g_bcg_ds.d_mapping, nodes);
    #endif

    // construct_BCG(g_bcg_ds, 0, true);

    auto new_dur = myTimer.stop();
    std::cout <<"computeBCG finished in: " << dur + new_dur <<" ms." << "\n";
}

