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

// Kernel to mark self-loops and duplicates
__global__ 
void markForRemoval(uint64_t* d_edges_input, unsigned char* flags, size_t num_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        uint64_t i = d_edges_input[idx];

        int edges_u = i >> 32;          // Extract higher 32 bits
        int edges_v = i & 0xFFFFFFFF;   // Extract lower 32 bits

        // Mark self-loops
        if (edges_u == edges_v) {
            flags[idx] = false;
        }
        // Mark duplicates (assuming edges are sorted)
        else if (idx > 0 && d_edges_input[idx] == d_edges_input[idx - 1]) {
            flags[idx] = false;
        }
        else {
            flags[idx] = true;
        }
    }
}

void select_flagged(
    uint64_t* d_in,             // input array
    uint64_t* d_out,            // output array
    unsigned char* d_flags,     // flag array denoting to select an edge or not
    long *h_num_selected_out,   // Selected number of edges (host variable)
    long *d_num_selected_out,   // Selected number of edges (device variable)
    long& num_items) {          // Selected number of edges (host variable)

    #ifdef DEBUG
        DisplayDeviceUint64Array(d_in, d_flags, num_items);
        DisplayDeviceUCharArray(d_flags, num_items);
    #endif
    
    // Allocate temporary storage
    size_t temp_storage_bytes = 0;

    DeviceSelect::Flagged(NULL, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    
    // Step 2: Allocate temporary storage
    void* d_temp_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate temporary storage for CUB");

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

    CUDA_CHECK(cudaMemcpy(h_num_selected_out, d_num_selected_out, sizeof(long), cudaMemcpyDeviceToHost),"Failed to copy back d_num_selected_out");
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");    

    CUDA_CHECK(cudaFree(d_temp_storage), "Failed to free temporary storage");    

    num_items = *h_num_selected_out;
    
    #ifdef DEBUG
        // Copy output data back to host
        uint64_t* h_out = new uint64_t[num_items];
        cudaMemcpy(h_out, d_out, sizeof(uint64_t) * num_items, cudaMemcpyDeviceToHost);

        // Print output data
        printf("\nOutput Data (h_out):\n");
        DisplayResults(h_out, h_num); // Print only the selected elements
    #endif
}

// Function to remove self-loops and duplicates from graph edges
void remove_self_loops_duplicates(
    uint64_t*&      d_edges_input,        // 1. Input edges
    long&           num_items,            // 2. Number of items (edges) in the input
    unsigned char*& d_flags,              // 3. Flags used to mark items for removal
    long*           h_num_selected_out,   // 4. Output: number of items selected (non-duplicates, non-self-loops) (host value)
    long*           d_num_selected_out,   // 5. Output: number of items selected (non-duplicates, non-self-loops) (device value)
    uint64_t*&      d_edges_output)       // 6. Output keys (processed edges' first vertices)
{
    // std::cout << "Inside remove_self_loops_duplicates.\n";
    if(g_verbose)
        std::cout <<"num_items before edge_cleanup: " << num_items << "\n";

    // Step 1: Determine the required temporary storage size
    size_t temp_storage_bytes = 0;
    cudaError_t status;
    status = cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_edges_input, d_edges_input, num_items);
    CUDA_CHECK(status, "Error in CUB SortKeys");

    // Step 2: Allocate temporary storage
    void* d_temp_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate temporary storage for CUB");

    // Step 3: Perform the sorting
    status = cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_edges_input, d_edges_input, num_items);
    CUDA_CHECK(status, "Error in CUB SortKeys");

    // Step 4: Free temporary storage
    CUDA_CHECK(cudaFree(d_temp_storage), "Failed to free temporary storage");

    // Optional: Synchronize if further operations depend on the sorted result
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after sorting");

    // Mark self-loops and duplicates for removal
    long numThreads = maxThreadsPerBlock;
    long numBlocks = (num_items + numThreads - 1) / numThreads;
    markForRemoval<<<numBlocks, numThreads>>>(d_edges_input, d_flags, num_items);
    CUDA_CHECK(cudaGetLastError(), "Error in markForRemoval kernel");

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");

    // std::cout << "Mark for removal kernel successfully finished." << std::endl;

    select_flagged(
    d_edges_input,              // input array
    d_edges_output,             // output array
    d_flags,                    // flag array denoting to select an edge or not
    h_num_selected_out,         // Selected number of edges (host variable)
    d_num_selected_out,         // Selected number of edges (device variable)
    num_items);                 // Output array size

    // std::cout << "select_flagged successfully finished." << std::endl;

    if(g_verbose) {
        std::cout << "Printing Edge List after cleaning up:" << std::endl;
        print_device_edges(d_edges_output, num_items);
        std::cout <<"NumEdges after cleaning up: " << num_items << "\n";
        std::cout <<"Cleaned edge stream:\n";
    }
}