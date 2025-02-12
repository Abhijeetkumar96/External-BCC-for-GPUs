/******************************************************************************
* Functionality: Remove Self-Loops and Duplicates
 ******************************************************************************/

#ifndef EDGE_CLEANUP_H
#define EDGE_CLEANUP_H

#include <cuda_runtime.h>

// Function to remove self-loops and duplicates from graph edges
void remove_self_loops_duplicates(
    uint64_t*&      d_edges_input,        // 1. Input edges
    long&           num_items,            // 2. Number of items (edges) in the input
    unsigned char*& d_flags,              // 3. Flags used to mark items for removal
    long*           h_num_selected_out,   // 4. Output: number of items selected (non-duplicates, non-self-loops) (host value)
    long*           d_num_selected_out,   // 5. Output: number of items selected (non-duplicates, non-self-loops) (device value)
    uint64_t*&      d_edges_output);      // 6. Output keys (processed edges' first vertices)

#endif // EDGE_CLEANUP_H