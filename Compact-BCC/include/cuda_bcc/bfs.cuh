#ifndef BFS_H
#define BFS_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utility/cuda_utility.cuh"

void constructSpanningTree(
    
    // Input variables
    int no_of_vertices,                     // Number of vertices in the graph
    long numEdges,                          // Number of edges in the graph
    long* d_offset,                         // Device pointer to offsets in adjacency list
    int* d_neighbours,                      // Device pointer to neighbors in adjacency list
    int* d_flag,                            // Device pointer to flag array (for processing)
    int* h_flag,                            // Host (pinned memory) pointer to flag array
    int root,                               // Root vertex for spanning tree construction
    
    // Output variables
    int* d_level,                           // Device pointer to levels of vertices in spanning tree
    int* d_parent,                          // Device pointer to parents of vertices in spanning tree
    int* d_child_of_root,                   // Device pointer to children count of root
    int* h_child_of_root,                   // Host (pinned memory) pointer to children count of root
    
    // Stream variables
    cudaStream_t computeStream,             // CUDA stream for computation
    cudaStream_t transD2HStream);           // CUDA stream for device-to-host transfers

#endif // BFS_H