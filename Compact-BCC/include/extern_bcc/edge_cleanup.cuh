/******************************************************************************
* Functionality: Remove Self-Loops and Duplicates
 ******************************************************************************/

#ifndef EDGE_CLEANUP_H
#define EDGE_CLEANUP_H

#include <cuda_runtime.h>

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
    cudaStream_t    computeStream,        // 11. 
    cudaStream_t    transD2HStream);      // 12. 

#endif // EDGE_CLEANUP_H