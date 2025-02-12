/******************************************************************************
* Functionality: Memory Management
* Handles allocation, deallocation, and initialization of all variables
 ******************************************************************************/

#ifndef BCC_MEM_UTILS_H
#define BCC_MEM_UTILS_H

#include <cuda_runtime.h>

class GPU_BCG {
public:
    GPU_BCG(int, long, long);
    ~GPU_BCG();

    void init(cudaStream_t myStream = 0);

    int numVert;
    long numEdges;
    long batchSize;
    long E;  // Double the edge count (e,g. for edge (2,3), (3,2) is also counted)
    long max_allot;

    int root;
    int child_of_root;

    const int orig_numVert;
    const long orig_numEdges;

    // CUDA streams
    cudaStream_t computeStream, transH2DStream, transD2HStream; // -> streams for computation and transfer

    // CUDA events
    cudaEvent_t event;

    /* Pointers for dynamic memory */
    // a. Edge stream
    const int* src;
    const int* dest;
    
    // b. device arrays
    // 1. Pointers for copy buffers
    int *d_edge_u[2];
    int *d_edge_v[2];

    // 2. input set of unique edges
    int *original_u, *original_v;

    // 3. duplicate edges for creating csr
    int *u_arr_buf, *v_arr_buf;

    // 4. Alternate buffers for sorting operation required for csr creation
    int *u_arr_alt_buf, *v_arr_alt_buf;

    // 5. vertex offset array for csr
    long *d_vertices;

    // 6. Spanning tree informations
    int* d_parent;
    int* d_level;
    int* d_org_parent;

    // 7. BCC-related parameters
    bool *d_isSafe, *d_isPartofFund;
    int  *d_cut_vertex;
    int  *d_baseVertex;     // Every non - tree edge has an associated base vertex
    bool *d_is_baseVertex;  // My parent is lca or not
    long *d_nonTreeEdgeId;
    int  *d_totalRootChild;
    int  *d_imp_bcc_num;    // Final Output

    // 8. Connected Components (CC) specific parameters
    int *d_baseU, *d_baseV, *d_rep;

    // 9. For updating bcc numbers
    int *d_bcc_flag = NULL;

    // 10. BCG related data-structures
    int  *d_mapping = NULL;
    bool *d_isFakeCutVertex = NULL;
    
    // 11. Device array for merged output
    int64_t *d_merged = NULL;

    // 12. common flag for both bfs and cc
    int *d_flag             = NULL;
    int *d_child_of_root    = NULL;

    // pinned memory
    int *h_flag             = NULL; // pinned memory
    long *h_num_items       = NULL; // pinned memory
    int *h_max_ps_bcc         = NULL; // pinned memory
    int *h_max_ps_cut_vertex  = NULL; // pinned memory
    int *h_child_of_root    = NULL; // pinned memory
    long *h_num_selected_out = NULL; // pinned memory

    // 13. temp storage to be used for all cub functions
    void *d_temp_storage        =  NULL;
    
    // 14. Stream Compaction variables
    unsigned char *d_flags      =  NULL;
    long *d_num_selected_out    =  NULL;
};

#endif // BCG_MEM_UTILS_H
