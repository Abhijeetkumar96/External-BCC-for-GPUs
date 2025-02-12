//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <iostream>   
#include <cstdint>    
#include <vector>    

//---------------------------------------------------------------------
// CUDA & CUB Libraries
//---------------------------------------------------------------------
#include <cuda_runtime.h> 
#include <cub/cub.cuh>

//---------------------------------------------------------------------
// CUDA Utility & helper functions
//---------------------------------------------------------------------
#include "extern_bcc/edge_cleanup.cuh" 
#include "utility/cuda_utility.cuh"

using namespace cub;

/****************************** Sorting starts ************************************/

// CUDA kernel to merge two integer arrays into an array of int64_t
__global__ 
void packPairs(const int *arrayU, const int *arrayV, int64_t *arrayE, long size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Cast to int64_t to ensure the shift operates on 64 bits
        int64_t u = arrayU[idx];
        int64_t v = arrayV[idx];
        // Ensure 'v' is treated as a 64-bit value
        arrayE[idx] = (u << 32) | (v & 0xFFFFFFFFLL);
    }
}

__global__ 
void unpackPairs(const int64_t *zippedArray, int *arrayA, int *arrayB, long size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Extract the upper 32 bits
        arrayA[idx] = zippedArray[idx] >> 32;
        // Extract the lower 32 bits, ensuring it's treated as a signed int
        arrayB[idx] = int(zippedArray[idx] & 0xFFFFFFFFLL);  
    }
}

// Validator function
void validateSortedPairs(int* d_keys, int* d_values, long num_items, cudaStream_t computeStream) {
    // Allocate host memory
    int* h_keys     = new int[num_items];
    int* h_values   = new int[num_items];

    // Copy sorted data back to host
    CUDA_CHECK(
        cudaMemcpyAsync(
            h_keys, 
            d_keys, 
            num_items * sizeof(int), 
            cudaMemcpyDeviceToHost, 
            computeStream
        ), 
        "Failed to copy keys to host"
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            h_values, 
            d_values, 
            num_items * sizeof(int), 
            cudaMemcpyDeviceToHost, 
            computeStream
        ), 
        "Failed to copy values to host"
    );
    // CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize after copying data to host");

    // Check if the keys are sorted
    bool isValid = true;
    for (long i = 1; i < num_items; i++) {
        if (h_keys[i-1] > h_keys[i]) {
            std::cerr << "Validation failed: Key array is not sorted at index " << i-1 << ". " << h_keys[i-1] << " > " << h_keys[i] << "\n";
            isValid = false;
            break;
        }
    }

    if (isValid) {
        std::cout << "Validation passed: Array is correctly sorted." << "\n";
    } else {
        std::cerr << "Validation failed: Array is not sorted correctly." << std::endl;
    }

    // Free host memory
    delete[] h_keys;
    delete[] h_values;
}


void radix_sort_for_pairs(
    int* d_keys, 
    int* d_values, 
    int64_t *d_merged, 
    long num_items, 
    void* d_temp_storage, 
    cudaStream_t computeStream ) {
    
    long threadsPerBlock = maxThreadsPerBlock;
    long blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
    
    if(g_verbose) {
        std::cout <<"Edge list before clean up: \n";
        kernelPrintEdgeList(d_keys, d_values, num_items, computeStream);
    }

    packPairs<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(d_keys, d_values, d_merged, num_items);
    CUDA_CHECK(cudaGetLastError(), "packPairs kernel launch failed");
    
    // Sort the packed pairs
    size_t temp_storage_bytes = 0;
    
    cudaError_t status;
    status = cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, d_merged, d_merged, num_items, 0, sizeof(int64_t)*8, computeStream);
    CUDA_CHECK(status, "Error in CUB SortKeys");

    status = cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_merged, d_merged, num_items, 0, sizeof(int64_t)*8, computeStream);
    CUDA_CHECK(status, "Error in CUB SortKeys");

    CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize stream after copying max prefix sums");

    // Kernel invocation for unpacking
    unpackPairs<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(d_merged, d_keys, d_values, num_items);
    CUDA_CHECK(cudaGetLastError(), "unpackPairs kernel launch failed");

    CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize stream after copying max prefix sums");

    if(g_verbose) {
        std::cout << "Edge list after clean up.\n";
        kernelPrintEdgeList(d_keys, d_values, num_items, computeStream);
    }

    if(checker) {
        validateSortedPairs(d_keys, d_values, num_items, computeStream);
    }
}

/****************************** Sorting ends ****************************************/

// Kernel to mark self-loops and duplicates
__global__ 
void markForRemoval(int* edges_u, int* edges_v, unsigned char* flags, size_t num_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        // Mark self-loops
        if (edges_u[idx] == edges_v[idx]) {
            flags[idx] = false;
        }
        // Mark duplicates (assuming edges are sorted)
        else if (idx > 0 && edges_u[idx] == edges_u[idx - 1] && edges_v[idx] == edges_v[idx - 1]) {
            flags[idx] = false;
        }
        else {
            flags[idx] = true;
        }
    }
}

// Function to remove self-loops and duplicates from graph edges
void remove_self_loops_duplicates(
    int*&           d_keys,               // 1. Input keys (edges' first vertices)
    int*&           d_values,             // 2. Input values (edges' second vertices)
    long&           num_items,            // 3. Number of items (edges) in the input
    int64_t*&       d_merged,             // 4. Intermediate storage for merged (zipped) keys and values
    unsigned char*& d_flags,              // 5. Flags used to mark items for removal
    long*           h_num_selected_out,   // 6. Output: number of items selected (non-duplicates, non-self-loops)
    long*           d_num_selected_out,   // 7. Output: number of items selected (non-duplicates, non-self-loops)
    int*&           d_keys_out,           // 8. Output keys (processed edges' first vertices)
    int*&           d_values_out,         // 9. Output values (processed edges' second vertices)
    void*&          d_temp_storage,       // 10. Temporary storage for intermediate computations
    cudaStream_t    computeStream,             // 11. 
    cudaStream_t    transD2HStream)       // 12. 

{
    std::cout <<"num_items before edge_cleanup: " << num_items << "\n";
    cudaError_t status;

    radix_sort_for_pairs(d_keys, d_values, d_merged, num_items, d_temp_storage, computeStream);

    // Mark self-loops and duplicates for removal
    long numThreads = maxThreadsPerBlock;
    long numBlocks = (num_items + numThreads - 1) / numThreads;
    markForRemoval<<<numBlocks, numThreads, 0, computeStream>>>(d_keys, d_values, d_flags, num_items);
    CUDA_CHECK(cudaGetLastError(), "Error in markForRemoval kernel");

    CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize stream after copying max prefix sums");

    size_t temp_storage_bytes = 0;

    // Remove marked edges
    // Determine temporary storage requirements for selection
    status = DeviceSelect::Flagged(NULL, temp_storage_bytes, d_keys, d_flags, d_keys_out, d_num_selected_out, num_items, computeStream);
    CUDA_CHECK(status, "Error in CUB Flagged");

    CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize stream after copying max prefix sums");
    
    // One call for keys
    status = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_keys, d_flags, d_keys_out, d_num_selected_out, num_items, computeStream);
    CUDA_CHECK(status, "Error in CUB Flagged");
    // One call for values
    status = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_values, d_flags, d_values_out, d_num_selected_out, num_items, computeStream);
    CUDA_CHECK(status, "Error in CUB Flagged");

    CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize stream after copying max prefix sums");

    CUDA_CHECK(cudaMemcpyAsync(h_num_selected_out, d_num_selected_out, sizeof(long), cudaMemcpyDeviceToHost, transD2HStream),"Failed to copy back d_num_selected_out");
    CUDA_CHECK(cudaStreamSynchronize(transD2HStream), "Failed to synchronize stream after copying max prefix sums");    
        
    num_items = *h_num_selected_out;

    if(g_verbose) {
        kernelPrintEdgeList(d_keys_out, d_values_out, num_items, computeStream);
        std::cout <<"NumEdges after cleaning up: " << num_items << "\n";
        std::cout <<"Cleaned edge stream:\n";
    }
}