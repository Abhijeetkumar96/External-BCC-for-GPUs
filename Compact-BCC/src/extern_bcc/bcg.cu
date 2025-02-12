//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <omp.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// BCC related functions
//---------------------------------------------------------------------
#include "cuda_bcc/bcc.cuh"
#include "cuda_bcc/lca.cuh"
#include "cuda_bcc/cut_vertex.cuh"

//---------------------------------------------------------------------
// BCG related functions
//---------------------------------------------------------------------
#include "extern_bcc/bcg.cuh"
#include "extern_bcc/edge_cleanup.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "utility/cuda_utility.cuh"

/*
  _____ _          _     ____        _       _       _                        _ 
 |  ___(_)_ __ ___| |_  | __ )  __ _| |_ ___| |__   | | _____ _ __ _ __   ___| |
 | |_  | | '__/ __| __| |  _ \ / _` | __/ __| '_ \  | |/ / _ \ '__| '_ \ / _ \ |
 |  _| | | |  \__ \ |_  | |_) | (_| | || (__| | | | |   <  __/ |  | | | |  __/ |
 |_|   |_|_|  |___/\__| |____/ \__,_|\__\___|_| |_| |_|\_\___|_|  |_| |_|\___|_|
                                                                                
*/
__global__ 
void update_initial_edges_kernel(
    const int* parent, const int* d_org_parent,                             // Parent info
    int* mapping, const int* bcc_num,                                       // Mapping info
    const int* nonTree_u,   const int* nonTree_v,                           // Zeroth batch non-tree edges
    const int* new_batch_U, const int* new_batch_V,                         // First batch, newly copied edges
    int* u_arr_buf, int* v_arr_buf,                                         // Output arrays
    const int parentSize, const int nonTreeSize, const int batchSize) {     // Sizes for each array


    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long totalSize = parentSize + nonTreeSize + batchSize;

    // Early exit for out-of-bounds threads
    if (idx >= totalSize) return;

    int u, v;
    
    // mapping tree edges
    if (idx < parentSize) {
        u = idx;
        v = parent[idx];
        mapping[idx] = bcc_num[idx]; // update the mapping

        u_arr_buf[idx] = min(bcc_num[u], bcc_num[v]);
        v_arr_buf[idx] = max(bcc_num[u], bcc_num[v]);
    }

    // mapping for the zeroth batch copied edges
    else if (idx < parentSize + nonTreeSize) {
        long nonTreeIdx = idx - parentSize;
        u = nonTree_u[nonTreeIdx];
        v = nonTree_v[nonTreeIdx];

        #ifdef DEBUG
            printf("u: %d, v: %d, idx: %d\n", u, v, idx);
        #endif

        if(d_org_parent[u] == v || d_org_parent[v] == u) {
            u_arr_buf[idx] = -1;
            v_arr_buf[idx] = -1;
        }
        else {
            u_arr_buf[idx] = min(bcc_num[u], bcc_num[v]);
            v_arr_buf[idx] = max(bcc_num[u], bcc_num[v]);
        }
    } 

    // new batch copied edges
    else {
        long batchIdx = idx - parentSize - nonTreeSize;
        u = new_batch_U[batchIdx];
        v = new_batch_V[batchIdx];
        
        #ifdef DEBUG
            printf("u: %d, v: %d, idx: %d\n", u, v, idx);
        #endif

        if(d_org_parent[u] == v || d_org_parent[v] == u) {
            u_arr_buf[idx] = -1;
            v_arr_buf[idx] = -1;
        }
        else {
            u_arr_buf[idx] = min(bcc_num[u], bcc_num[v]);
            v_arr_buf[idx] = max(bcc_num[u], bcc_num[v]);
        }
    }
}

/*
  ____                      _       _               _           _       _                 _                        _ 
 |  _ \ ___ _ __ ___   __ _(_)_ __ (_)_ __   __ _  | |__   __ _| |_ ___| |__   ___  ___  | | _____ _ __ _ __   ___| |
 | |_) / _ \ '_ ` _ \ / _` | | '_ \| | '_ \ / _` | | '_ \ / _` | __/ __| '_ \ / _ \/ __| | |/ / _ \ '__| '_ \ / _ \ |
 |  _ <  __/ | | | | | (_| | | | | | | | | | (_| | | |_) | (_| | || (__| | | |  __/\__ \ |   <  __/ |  | | | |  __/ |
 |_| \_\___|_| |_| |_|\__,_|_|_| |_|_|_| |_|\__, | |_.__/ \__,_|\__\___|_| |_|\___||___/ |_|\_\___|_|  |_| |_|\___|_|
                                            |___/                                                                    
*/
__global__ 
void update_remaining_edges_kernel(
    int* bcc_num, int* d_mapping, const int* d_org_parent,  // Mapping info
    int* d_prev_batch_U, int* d_prev_batch_V,               // Previous Graph edges
    int* d_new_batch_U, int* d_new_batch_V,                 // Newly copied edges
    int* u_arr_buf, int* v_arr_buf,                         // Output arrays
    long prev_num_edges, long totalSize) {                  // Sizes for each array

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= totalSize) return;
    
    if(idx < prev_num_edges) {
        int u = d_prev_batch_U[idx];
        int v = d_prev_batch_V[idx];
        
        #ifdef DEBUG
            printf("u: %d, v: %d, idx: %d\n", u, v, idx);
        #endif
        
        u_arr_buf[idx] = min(bcc_num[u], bcc_num[v]);
        v_arr_buf[idx] = max(bcc_num[u], bcc_num[v]);

    } else {
        int u = d_new_batch_U[idx - prev_num_edges];
        int v = d_new_batch_V[idx - prev_num_edges];
        
        #ifdef DEBUG
            printf("u: %d, v: %d, idx: %d\n", u, v, idx);
        #endif

        if (d_org_parent[u] == v || d_org_parent[v] == u) {
            u_arr_buf[idx] = -1;
            v_arr_buf[idx] = -1;
        } else {
            int mapped_u = d_mapping[u];
            int mapped_v = d_mapping[v];

            u_arr_buf[idx] = min(bcc_num[mapped_u], bcc_num[mapped_v]);
            v_arr_buf[idx] = max(bcc_num[mapped_u], bcc_num[mapped_v]);
        }
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

/*
  ____                                _____              _   _       ____        _       _     
 |  _ \ _ __ ___   ___ ___  ___ ___  |__  /___ _ __ ___ | |_| |__   | __ )  __ _| |_ ___| |__  
 | |_) | '__/ _ \ / __/ _ \/ __/ __|   / // _ \ '__/ _ \| __| '_ \  |  _ \ / _` | __/ __| '_ \ 
 |  __/| | | (_) | (_|  __/\__ \__ \  / /|  __/ | | (_) | |_| | | | | |_) | (_| | || (__| | | |
 |_|   |_|  \___/ \___\___||___/___/ /____\___|_|  \___/ \__|_| |_| |____/ \__,_|\__\___|_| |_|
                                                                                               
*/
void process_zeroth_batch(
    GPU_BCG& g_bcg_ds, 
    size_t batchSize,
    bool isLastBatch = false) {
    
    std::cout <<"\n\t***********processing zeroth batch.***********\n" << std::endl;

    int numVert       =  g_bcg_ds.numVert;
    g_bcg_ds.numEdges =  batchSize;

    int* h_child_of_root = g_bcg_ds.h_child_of_root;

    naive_lca(g_bcg_ds, g_bcg_ds.root);
    assign_cut_vertex_BCC(g_bcg_ds, g_bcg_ds.root, *h_child_of_root, true, false);

    if(!isLastBatch)
        g_bcg_ds.numVert = update_bcc_numbers(g_bcg_ds, numVert);

    int* d_cut_vertex = g_bcg_ds.d_cut_vertex;
    int* d_imp_bcc_num = g_bcg_ds.d_imp_bcc_num;

    if(g_verbose) {
        
        std::cout << "isLastBatch: " << isLastBatch << std::endl;
        std::cout << "cut_vertex status:" << std::endl;
        kernelPrintArray(d_cut_vertex, numVert, g_bcg_ds.computeStream);
        std::cout << "ibcc numbers:" << std::endl;
        kernelPrintArray(d_imp_bcc_num, numVert, g_bcg_ds.computeStream);
    }
}

/*
  ____                                 __ _          _     ____        _       _     
 |  _ \ _ __ ___   ___ ___  ___ ___   / _(_)_ __ ___| |_  | __ )  __ _| |_ ___| |__  
 | |_) | '__/ _ \ / __/ _ \/ __/ __| | |_| | '__/ __| __| |  _ \ / _` | __/ __| '_ \ 
 |  __/| | | (_) | (_|  __/\__ \__ \ |  _| | |  \__ \ |_  | |_) | (_| | || (__| | | |
 |_|   |_|  \___/ \___\___||___/___/ |_| |_|_|  |___/\__| |____/ \__,_|\__\___|_| |_|
                                                                                     
*/
void process_first_batch(
    GPU_BCG& g_bcg_ds, 
    size_t zeroBatchSize, 
    size_t firstBatchSize, 
    const int& original_numVert,
    bool isLastBatch) 
{

    std::cout <<"\n\t*********** processing first batch. ***********\n";    

    long totalSize = original_numVert + zeroBatchSize + firstBatchSize;

    std::cout <<"totalSize: " << totalSize << std::endl;
    assert(totalSize <= g_bcg_ds.max_allot);

    long threadsPerBlock = 1024;
    size_t blocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Printing From process first batch:\n";

    update_initial_edges_kernel<<<blocks, threadsPerBlock, 0, g_bcg_ds.computeStream>>>(
        g_bcg_ds.d_parent,      // input
        g_bcg_ds.d_org_parent,  // input
        g_bcg_ds.d_mapping,     // input (actual_mapping)
        g_bcg_ds.d_imp_bcc_num, // input (bcc_number)
        g_bcg_ds.original_u,    // input
        g_bcg_ds.original_v,    // input                                     
        g_bcg_ds.d_edge_u[0],   // input (d_new_batch_U)
        g_bcg_ds.d_edge_v[0],   // input (d_new_batch_V)
        g_bcg_ds.u_arr_buf,     // output
        g_bcg_ds.v_arr_buf,     // output
        original_numVert,       // sizes
        zeroBatchSize,          // sizes
        firstBatchSize          // sizes
    );

    CUDA_CHECK(cudaGetLastError(), "Error in markForRemoval kernel");
    // CUDA_CHECK(cudaStreamSynchronize(myStream), "Failed to synchronize after markForRemoval");

    // do clean-up, i.e. remove self loops and duplicates.
    remove_self_loops_duplicates(
        g_bcg_ds.u_arr_buf,           // 1. Input edge-stream U
        g_bcg_ds.v_arr_buf,           // 2. Input edge-stream V
        totalSize,                    // 3. Total number of items (input & output)
        g_bcg_ds.d_merged,            // 4. Intermediate merged storage
        g_bcg_ds.d_flags,             // 5. Flags for marking
        g_bcg_ds.h_num_selected_out,  // 6. Output: selected count
        g_bcg_ds.d_num_selected_out,  // 7. Output: selected count
        g_bcg_ds.original_u,          // 8. Output keys
        g_bcg_ds.original_v,          // 9. Output values
        g_bcg_ds.d_temp_storage,      // 10. Temporary storage
        g_bcg_ds.computeStream,       // 11. 
        g_bcg_ds.transD2HStream       // 12. 
    );

    // total size contains the remaining edgeCount after clean-up
    // In next release, make everything device (no need to copy back numVerts nd numEdges)
    g_bcg_ds.numEdges = totalSize;

    // Check if at anytime the alloted numEdges is becoming less than the current numEdges
    std::cout <<"g_bcg_ds.numEdges: " << g_bcg_ds.numEdges << std::endl;
    std::cout <<"g_bcg_ds.max_allot: " << g_bcg_ds.max_allot << std::endl;
    assert(g_bcg_ds.numEdges <= g_bcg_ds.max_allot);

    // call cuda_bcc
    std::cout << "edge_cleanup over, calling cuda_bcc.\n";
    cuda_bcc(g_bcg_ds, isLastBatch);
}

 /*
  ____                                ____                      _       _               ____        _       _     
 |  _ \ _ __ ___   ___ ___  ___ ___  |  _ \ ___ _ __ ___   __ _(_)_ __ (_)_ __   __ _  | __ )  __ _| |_ ___| |__  
 | |_) | '__/ _ \ / __/ _ \/ __/ __| | |_) / _ \ '_ ` _ \ / _` | | '_ \| | '_ \ / _` | |  _ \ / _` | __/ __| '_ \ 
 |  __/| | | (_) | (_|  __/\__ \__ \ |  _ <  __/ | | | | | (_| | | | | | | | | | (_| | | |_) | (_| | || (__| | | |
 |_|   |_|  \___/ \___\___||___/___/ |_| \_\___|_| |_| |_|\__,_|_|_| |_|_|_| |_|\__, | |____/ \__,_|\__\___|_| |_|
                                                                                |___/                             
*/
void process_remaining_batches(
    GPU_BCG&        g_bcg_ds,           
    const size_t&   BatchSize,           // Size of the current batch
    const int&      original_num_verts,  // Original number of vertices
    const int&     bufferIndex,         // Copy buffer index
    const bool&     isLastBatch) 
{

    int*      d_prev_batch_U    =   g_bcg_ds.original_u;
    int*      d_prev_batch_V    =   g_bcg_ds.original_v;
    int*      d_new_batch_U     =   g_bcg_ds.d_edge_u[bufferIndex];
    int*      d_new_batch_V     =   g_bcg_ds.d_edge_v[bufferIndex];

    // old_number_of_edges + new_batch_size
    long totalSize = g_bcg_ds.numEdges + BatchSize;
    std::cout << "bufferIndex from process_remaining_batches: " << bufferIndex << std::endl;
    std::cout <<"Edge Count after adding new batch / totalSize: " << totalSize << std::endl;
    assert(totalSize <= g_bcg_ds.max_allot);
    
    long threadsPerBlock = maxThreadsPerBlock;

    size_t blocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    update_remaining_edges_kernel<<<blocks, threadsPerBlock, 0, g_bcg_ds.computeStream>>>(
        g_bcg_ds.d_imp_bcc_num,    // BCC numbers
        g_bcg_ds.d_mapping,        // Vertex ID mapping
        g_bcg_ds.d_org_parent,     // Original parent array
        d_prev_batch_U,            // Previous batch_U
        d_prev_batch_V,            // Previous batch_V
        d_new_batch_U,             // New batch U vertices
        d_new_batch_V,             // New batch V vertices
        g_bcg_ds.u_arr_buf,        // Output after remapping for U vertices
        g_bcg_ds.v_arr_buf,        // Output after remapping for V vertices
        g_bcg_ds.numEdges,         
        totalSize
    );
    // CUDA_CHECK(cudaStreamSynchronize(myStream), "Failed to synchronize update_remaining_edges_kernel");
    blocks = (original_num_verts + threadsPerBlock - 1) / threadsPerBlock;
    
    // Update the actual vertex mapping array for the original graph
    update_mapping<<<blocks, threadsPerBlock, 0, g_bcg_ds.computeStream>>>(
        g_bcg_ds.d_mapping,      // Vertex ID mapping
        g_bcg_ds.d_imp_bcc_num,  // Important BCC numbers
        original_num_verts       // Original number of vertices
    );
    // CUDA_CHECK(cudaStreamSynchronize(myStream), "Failed to synchronize update_mapping");
   
    // do clean-up, i.e. remove self loops and duplicates.
    remove_self_loops_duplicates(
        g_bcg_ds.u_arr_buf,           // Input edge-stream U
        g_bcg_ds.v_arr_buf,           // Input edge-stream V
        totalSize,                    // Total number of items
        g_bcg_ds.d_merged,            // Intermediate merged storage
        g_bcg_ds.d_flags,             // Flags for marking
        g_bcg_ds.h_num_selected_out,  // Output: selected count
        g_bcg_ds.d_num_selected_out,  // Output: selected count
        g_bcg_ds.original_u,          // Output keys
        g_bcg_ds.original_v,          // Output values
        g_bcg_ds.d_temp_storage,      // Temporary storage
        g_bcg_ds.computeStream,       // CUDA stream (optional)
        g_bcg_ds.transD2HStream
    );

    g_bcg_ds.numEdges = totalSize;
    
    // Check if at anytime the alloted numEdges is becoming less than the current numEdges
    assert(g_bcg_ds.numEdges <= g_bcg_ds.max_allot); 

    // call cuda_bcc
    std::cout << "Calling cuda_bcc\n";
    cuda_bcc(g_bcg_ds, isLastBatch);
}


/* ____                            _         ____   ____ ____ 
  / ___|___  _ __ ___  _ __  _   _| |_ ___  | __ ) / ___/ ___|
 | |   / _ \| '_ ` _ \| '_ \| | | | __/ _ \ |  _ \| |  | |  _ 
 | |__| (_) | | | | | | |_) | |_| | ||  __/ | |_) | |__| |_| |
  \____\___/|_| |_| |_| .__/ \__,_|\__\___| |____/ \____\____|
                      |_|                                     
*/

void computeBCG(GPU_BCG& g_bcg_ds) {

    int numVert                     =   g_bcg_ds.numVert;
    int original_numVert            =   numVert;
    long numEdges                   =   g_bcg_ds.orig_numEdges;
    long batchSize                  =   g_bcg_ds.batchSize;  
    cudaStream_t computeStream      =   g_bcg_ds.computeStream;
    cudaStream_t transH2DStream     =   g_bcg_ds.transH2DStream;

    // host nonTree edge-stream (pinned memory)
    const int *h_nonTreeEdges_u = g_bcg_ds.src;
    const int *h_nonTreeEdges_v = g_bcg_ds.dest;

    // Calculate the total number of batches
    int numBatches = (numEdges + batchSize - 1) / batchSize;
    std::cout << "Number of batches: " << numBatches << "\n";

    int zeroBatchSize = std::min(batchSize, numEdges);
    size_t bytes = zeroBatchSize * sizeof(int);
    
    std::cout << "Copying batch 0" << " to GPU (0 " << "to " << zeroBatchSize - 1 << ") on buffer_0.\n";
    
    //---------------------------------------------------------------------
    // Copying Zeroth Batch to GPU 
    //---------------------------------------------------------------------
    CUDA_CHECK(
        cudaMemcpyAsync(
            g_bcg_ds.original_u,                                // Destination pointer
            h_nonTreeEdges_u,                                   // Source pointer
            bytes,                                              // Number of bytes to copy
            cudaMemcpyHostToDevice,                             // Direction of copy 
            transH2DStream                                      // CUDA stream to use for this operation
        ),
        "Failed to copy zeroth batch of edges (u) to device"    // error message   
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            g_bcg_ds.original_v,                                // Destination pointer
            h_nonTreeEdges_v,                                   // Source pointer
            bytes,                                              // Number of bytes to copy
            cudaMemcpyHostToDevice,                             // Direction of copy 
            transH2DStream                                      // CUDA stream to use for this operation
        ),
        "Failed to copy zeroth batch of edges (v) to device"    // error message
    );
    CUDA_CHECK(cudaStreamSynchronize(transH2DStream), "Failed to synchronize transH2DStream stream");

    bool isLastBatch = false;
    long lastBatchSize = 0;

    if(numBatches == 1) {
        std::cout <<"Processing zeroth batch (zeorth batch is last Batch) \n";
        process_zeroth_batch(g_bcg_ds, zeroBatchSize, true);
        std::cout <<"Exiting..\n";
        return;
    }

    else if(numBatches == 2) {
        long remaining_edges = numEdges - batchSize;
        long zeroBatchSize = batchSize;
        long firstBatchSize = remaining_edges;
        isLastBatch = true;
        std::cout << "Copying batch 1" << " consisting of " << remaining_edges << " to GPU (edges " << batchSize << " to " << batchSize + remaining_edges << ") on buffer_1.\n";
        bytes = remaining_edges * sizeof(int);

        //---------------------------------------------------------------------
        // Copying first Batch in the meanwhile
        //---------------------------------------------------------------------
        CUDA_CHECK(
            cudaMemcpyAsync(
                g_bcg_ds.d_edge_u[0], 
                h_nonTreeEdges_u + zeroBatchSize, 
                bytes, 
                cudaMemcpyHostToDevice, 
                transH2DStream
            ), 
            "Failed to copy first batch of edges (u) to device"
        );

        CUDA_CHECK(
            cudaMemcpyAsync(
                g_bcg_ds.d_edge_v[0], 
                h_nonTreeEdges_v + zeroBatchSize, 
                bytes, 
                cudaMemcpyHostToDevice, 
                transH2DStream
            ), 
            "Failed to copy first batch of edges (v) to device"
        );

        std::cout << "Processing zeroth_batch.\n";
        process_zeroth_batch(g_bcg_ds, batchSize);
        std::cout << "wait for completion of processing of zeroth_batch and copy of first_batch.\n";
        CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize transH2DStream stream");
        std::cout <<"processing first batch (first batch is last Batch) \n";
        process_first_batch(g_bcg_ds, zeroBatchSize, firstBatchSize, original_numVert, isLastBatch);
        CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize transH2DStream stream");
        std::cout <<"Exiting..\n";
        return;
    } 

    else {
        for(int i = 0; i < numBatches - 1; ++i) {
            int startEdge = (i+1) * batchSize;
            int endEdge = std::min((i + 2) * batchSize - 1, numEdges - 1); // Adjust for last batch
            int edgesInBatch = endEdge - startEdge + 1;
            bool bufferIndex = i % 2;
            
            bytes = edgesInBatch * sizeof(int); 
            
            std::cout << "Copying batch " << i + 1 << " to GPU (edges " << startEdge << " to " << endEdge << ").\n";
            std::cout << "bufferIndex: " << bufferIndex << std::endl;
            
            //---------------------------------------------------------------------
            // Copying Batches
            //---------------------------------------------------------------------

            CUDA_CHECK(
                cudaMemcpyAsync(
                    g_bcg_ds.d_edge_u[bufferIndex], 
                    h_nonTreeEdges_u + (i + 1) * batchSize, 
                    bytes, 
                    cudaMemcpyHostToDevice, 
                    transH2DStream
                ), 
                "Failed to copy batch of edges (u) to device"
            );

            CUDA_CHECK(
                cudaMemcpyAsync(
                    g_bcg_ds.d_edge_v[bufferIndex], 
                    h_nonTreeEdges_v + (i + 1) * batchSize, 
                    bytes, 
                    cudaMemcpyHostToDevice, 
                    transH2DStream
                ), 
                "Failed to copy batch of edges (v) to device"
            );

            if(i == 0) {
                std::cout <<"Processing zeroth batch\n";
                std::cout << "bufferIndex: " << bufferIndex << std::endl;
                process_zeroth_batch(g_bcg_ds, batchSize);
            } 

            else if(i == 1) {
                // wait for the computation of zeroth batch to complete
                // wait for the copy of previous batch to complete
                CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize transH2DStream stream");
                std::cout <<"processing first batch\n ";
                process_first_batch(g_bcg_ds, batchSize, batchSize, original_numVert, false);
            } 

            else {
                isLastBatch = false;
                // wait for the computation of last batch to complete
                // wait for the copy of previous batch to complete
                std::cout <<"*********** Processing " << i << " Batch. ***********\n";
                // process_remaining_batches(g_bcg_ds, batchSize, original_numVert, bufferIndex, isLastBatch);
                process_remaining_batches(g_bcg_ds, batchSize, original_numVert, (bufferIndex + 1)%2, isLastBatch);
            }
            lastBatchSize = edgesInBatch;
        }
        //---------------------------------------------------------------------
        // Last Batch Processing
        //---------------------------------------------------------------------
        std::cout <<"*********** Processing last batch. ***********\n";
        isLastBatch = true;
        // bool bufferIndex = (numBatches - 1) % 2;
        bool bufferIndex = numBatches % 2;
        process_remaining_batches(g_bcg_ds, lastBatchSize, original_numVert, bufferIndex, isLastBatch);
    }
}

// ====[ End of computeBCG Code ]====