//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "utility/timer.hpp"
#include "utility/utility.hpp"
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

//---------------------------------------------------------------------
// BCC Specific Utilities
//---------------------------------------------------------------------
#include "cuda_bcc/lca.cuh"
#include "cuda_bcc/connected_components.cuh"


//---------------------------------------------------------------------
// LCA Kernel
//---------------------------------------------------------------------

__global__
void find_LCA(
    long num_non_tree_edges, 
    int* non_tree_u, int* non_tree_v, 
    int* parent, int* distance, 
    bool* is_marked, bool* is_safe, 
    long* nonTreeId, 
    int* base_u, int* base_v, 
    int* baseVertex, bool* d_isBaseVertex) {

    long i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < num_non_tree_edges) {
        int to = non_tree_u[i];
        int from = non_tree_v[i];

        // if a tree edge, update base vertices to 0.
        if(parent[to] == from || parent[from] == to) {
            base_u[i] = 0;
            base_v[i] = 0;

            baseVertex[i] = -1;
            return;
        }

        int higher = distance[to] < distance[from] ? to : from;
        int lower = higher == to ? from : to;
        int diff = distance[lower] - distance[higher];

        // Equalize heights
        while (diff--) {

            is_safe[lower] = true;
            is_marked[lower] = true;
            nonTreeId[lower] = i;

            lower = parent[lower];
            is_marked[lower] = 1;
        }

        // Mark till LCA is found
        while (parent[lower] != parent[higher]) {

            is_safe[lower] = 1;
            is_marked[lower] = 1;
            nonTreeId[lower] = i;

            lower = parent[lower];

            is_safe[higher] = 1;
            is_marked[higher] = 1;
            nonTreeId[higher] = i;

            higher = parent[higher];
        }

        // Update base vertices
        base_u[i] = lower;
        base_v[i] = higher;

        d_isBaseVertex[lower]   = true;
        d_isBaseVertex[higher]  = true;

        is_marked[lower]  = true;
        is_marked[higher] = true;
        
        nonTreeId[lower] = i;
        nonTreeId[higher] = i;

        baseVertex[i] = lower;
    }
}

void naive_lca(GPU_BCG& g_bcg_ds, int root) {

    //------------------------- Inputs -------------------------------
    int numVert            =    g_bcg_ds.numVert;
    long numNonTreeEdges   =    g_bcg_ds.numEdges;

    int *d_level           =    g_bcg_ds.d_level;
    int *d_parent          =    g_bcg_ds.d_parent;
    int *d_nonTreeEdge_U   =    g_bcg_ds.original_u;
    int *d_nonTreeEdge_V   =    g_bcg_ds.original_v;
    //-----------------------------------------------------------------
    
    //------------------------- Outputs -------------------------------
    
    //--- 1. For Connected Component Output -----
    int *d_rep             =    g_bcg_ds.d_rep;
    int *d_baseU           =    g_bcg_ds.d_baseU;
    int *d_baseV           =    g_bcg_ds.d_baseV;
    
    //--- 2. For LCA and Related Information ---
    long *d_nonTreeEdgeId  =    g_bcg_ds.d_nonTreeEdgeId; 
    
    bool *d_isSafe         =    g_bcg_ds.d_isSafe; 
    bool *d_isPartofFund   =    g_bcg_ds.d_isPartofFund; 

    // d_isBaseVertex:- Am I a base vertex, i.e. is my parent a lca vertex
    bool *d_is_baseVertex  =    g_bcg_ds.d_is_baseVertex; 
    
    // d_baseVertex:- Every non - tree edge has an associated base vertex
    int *d_baseVertex      =    g_bcg_ds.d_baseVertex; 

    //-----------------------------------------------------------------
    
    const long threadsPerBlock = maxThreadsPerBlock;
    long numBlocksEdges = (numNonTreeEdges + threadsPerBlock - 1) / threadsPerBlock;
    
    if(g_verbose)
        kernelPrintEdgeList(d_nonTreeEdge_U, d_nonTreeEdge_V, numNonTreeEdges, g_bcg_ds.computeStream);

    find_LCA<<<numBlocksEdges, threadsPerBlock, 0, g_bcg_ds.computeStream>>>(
        numNonTreeEdges,     // Input
        d_nonTreeEdge_U,     // Input
        d_nonTreeEdge_V,     // Input
        d_parent,            // Input
        d_level,             // Input

        d_isPartofFund,      // Output
        d_isSafe,            // Output
        d_nonTreeEdgeId,     // Output
        d_baseU,             // Output
        d_baseV,             // Output
        d_baseVertex,        // Output
        d_is_baseVertex      // Output
    );

    CUDA_CHECK(cudaGetLastError(), "find_LCA Kernel launch failed");

    // call cc 
        // call cc 
    connected_comp(
        numNonTreeEdges,    // 1
        d_baseU,            // 2
        d_baseV,            // 3
        numVert,            // 4
        d_rep,              // 5
        g_bcg_ds.h_flag,             // 6
        g_bcg_ds.d_flag,             // 7
        g_bcg_ds.computeStream,   // 8
        g_bcg_ds.transD2HStream
    );
}

// ====[ End of naive_lca Code ]====