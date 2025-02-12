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


__global__ 
void init_kernel(
    int *d_level, int *d_bcc_flag, int *d_cut_vertex, int *d_baseVertex,
    bool *d_isSafe, bool *d_isPartofFund, bool *d_is_baseVertex, bool *d_isFakeCutVertex, 
    long *d_nonTreeEdgeId, long numVert, long numEdges) {
    
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numEdges) return;
    
    if (idx < numVert) {
        d_level[idx]            =   -1;
        d_baseVertex[idx]       =   -1;
        d_nonTreeEdgeId[idx]    =   -1;

        d_isSafe[idx]           =   false;
        d_bcc_flag[idx]         =   0;
        d_cut_vertex[idx]       =   0;
        d_isPartofFund[idx]     =   false;
        d_is_baseVertex[idx]    =   false;
        d_isFakeCutVertex[idx]  =   false;
    }
    if(idx >= numVert)
        d_baseVertex[idx] = -1;
}

// Function to allocate temporary storage for cub functions
size_t AllocateTempStorage(void** d_temp_storage, long num_items) {
    size_t temp_storage_bytes = 0;
    size_t required_bytes = 0;

    // Determine the temporary storage requirement for DeviceRadixSort::SortPairs
    cub::DeviceRadixSort::SortPairs(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Determine the temporary storage requirement for DeviceScan::InclusiveSum
    cub::DeviceScan::InclusiveSum(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Determine the temporary storage requirement for DeviceSelect::Flagged
    cub::DeviceSelect::Flagged(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Allocate the maximum required temporary storage
    CUDA_CHECK(cudaMalloc(d_temp_storage, temp_storage_bytes), "cudaMalloc failed for temporary storage for CUB operations");

    return temp_storage_bytes;
}

GPU_BCG::GPU_BCG(int vertices, long num_edges, long _batchSize) : numVert(vertices), orig_numVert(vertices), orig_numEdges(num_edges), batchSize(_batchSize) {

     // do the changes here for maximum possible number of edges
    // numEdges = (2 * batchSize) + (2 * numVert);
    numEdges = (2 * batchSize) + (4 * numVert);
    max_allot = numEdges;
    E = 2 * numEdges; // Two times the original edges count
    
    size_t size            = E        * sizeof(int);
    size_t bool_vert_size  = numVert  * sizeof(bool);
    size_t int_vert_size   = numVert  * sizeof(int);
    size_t long_vert_size  = numVert  * sizeof(long);
    
    Timer myTimer;
    
    // Create CUDA streams
    // -> stream for computation
    CUDA_CHECK(cudaStreamCreate(&computeStream), "Failed to create computeStream");
    // -> stream for Host to Device transfer
    CUDA_CHECK(cudaStreamCreate(&transH2DStream), "Failed to create transH2DStream");
    // -> stream for Device to Host transfer
    CUDA_CHECK(cudaStreamCreate(&transD2HStream), "Failed to create transD2HStream");

    // Create CUDA events for synchronization
    CUDA_CHECK(cudaEventCreate(&event), "Failed to create cudaEvent");

    // Function to allocate copy buffers
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc((void**)&d_edge_u[i], batchSize * sizeof(int)), "Failed to allocate copy buffer_u");
        CUDA_CHECK(cudaMalloc((void**)&d_edge_v[i], batchSize * sizeof(int)), "Failed to allocate copy buffer_v");
    }

    // Allocate memory for original edges 
    CUDA_CHECK(cudaMalloc((void**)&original_u, numEdges * sizeof(int)),     "Failed to allocate original_u array");
    CUDA_CHECK(cudaMalloc((void**)&original_v, numEdges * sizeof(int)),     "Failed to allocate original_v array");
    
    // Allocate memory for duplicate edges for creating csr
    CUDA_CHECK(cudaMalloc(&u_arr_buf, size),                                "Failed to allocate u_arr array.");
    CUDA_CHECK(cudaMalloc(&v_arr_buf, size),                                "Failed to allocate v_arr array.");

    // Allocate alternate buffers for sorting operation
    CUDA_CHECK(cudaMalloc(&u_arr_alt_buf, size),                            "Failed to allocate u_arr_alt_buf array.");
    CUDA_CHECK(cudaMalloc(&v_arr_alt_buf, size),                            "Failed to allocate v_arr_alt_buf array.");

    // Allocate vertex offset array for csr 
    CUDA_CHECK(cudaMalloc(&d_vertices, (numVert + 1) * sizeof(long)),       "Failed to allocate vertices array.");

    // Allocate arrays for spanning tree output
    CUDA_CHECK(cudaMalloc(&d_parent, int_vert_size),                        "Failed to allocate parent array.");
    CUDA_CHECK(cudaMalloc(&d_level,  int_vert_size),                        "Failed to allocate level array.");
    CUDA_CHECK(cudaMalloc(&d_org_parent, int_vert_size),                    "Failed to allocate parent array.");
    CUDA_CHECK(cudaMalloc(&d_child_of_root, sizeof(int)),                   "Failed to allocate d_child_of_root value");

    // Allocate memory for LCA traversal (cudaBCC)
    CUDA_CHECK(cudaMalloc(&d_isSafe,            bool_vert_size),            "Failed to allocate memory for d_isSafe");
    CUDA_CHECK(cudaMalloc(&d_cut_vertex,        int_vert_size),             "Failed to allocate memory for d_cut_vertex");
    CUDA_CHECK(cudaMalloc(&d_imp_bcc_num,       int_vert_size),             "Failed to allocate memory for d_imp_bcc_num");
    CUDA_CHECK(cudaMalloc(&d_isPartofFund,      bool_vert_size),            "Failed to allocate memory for d_isPartofFund");
    CUDA_CHECK(cudaMalloc(&d_is_baseVertex,     bool_vert_size),            "Failed to allocate memory for d_is_baseVertex");
    CUDA_CHECK(cudaMalloc(&d_nonTreeEdgeId,     long_vert_size),            "Failed to allocate memory for d_nonTreeEdgeId");

    CUDA_CHECK(cudaMalloc(&d_baseVertex, numEdges * sizeof(int)),           "Failed to allocate memory for d_baseVertex");

    // For connected Components
    CUDA_CHECK(cudaMalloc(&d_rep,        int_vert_size),                    "Failed to allocate memory for d_rep");
    CUDA_CHECK(cudaMalloc(&d_baseU,      numEdges * sizeof(int)),           "Failed to allocate memory for d_baseU");
    CUDA_CHECK(cudaMalloc(&d_baseV,      numEdges * sizeof(int)),           "Failed to allocate memory for d_baseV");

    // For updating bcc numbers
    CUDA_CHECK(cudaMalloc(&d_bcc_flag,  int_vert_size),                     "Failed to allocate memory for d_bcc_flag");
    
    // BCG DS
    CUDA_CHECK(cudaMalloc(&d_mapping,         int_vert_size),               "Failed to allocate memory for d_mapping");
    CUDA_CHECK(cudaMalloc(&d_isFakeCutVertex, bool_vert_size),              "Failed to allocate memory for d_isFakeCutVertex");
    CUDA_CHECK(cudaMalloc((void**)&d_merged,  numEdges * sizeof(int64_t)),  "Failed to allocate memory for d_isFakeCutVertex");
    
    // Common flag for bfs and cc
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)),                            "Failed to allocate flag value");

    // pinned memories
    CUDA_CHECK(cudaMallocHost(&h_flag, sizeof(int)),                        "Failed to allocate pinned memory for flag value");
    CUDA_CHECK(cudaMallocHost(&h_max_ps_bcc, sizeof(int)),                    "Failed to allocate pinned memory for max_ps_bcc value");
    CUDA_CHECK(cudaMallocHost(&h_max_ps_cut_vertex, sizeof(int)),             "Failed to allocate pinned memory for max_ps_cut_vertex value");
    CUDA_CHECK(cudaMallocHost(&h_child_of_root, sizeof(int)),               "Failed to allocate pinned memory for h_child_of_root value");

    CUDA_CHECK(cudaMallocHost(&h_num_selected_out, sizeof(long)),                  "Failed to allocate pinned memory for h_num_items value");
    CUDA_CHECK(cudaMallocHost(&h_num_items, sizeof(long)),                  "Failed to allocate pinned memory for h_num_items value");


    // Allocate memory streamCompaction
    CUDA_CHECK(cudaMalloc((void**)&d_flags, numEdges * sizeof(unsigned char)),
        "Failed to allocate flag array");
    
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(long)),
        "Failed to allocate d_num_selected_out");

    // Determine temporary storage requirements and allocate
    auto temp_s = AllocateTempStorage(&d_temp_storage, E);

    // Initialize GPU memory
    init();
    auto dur = myTimer.stop();

    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte), "Error: cudaMemGetInfo fails");

    double free_db = static_cast<double>(free_byte); 
    double total_db = static_cast<double>(total_byte);
    double used_db = total_db - free_db;

    std::cout << "========================================\n"
          << "       Allocation Details & GPU Memory Usage\n"
          << "========================================\n\n"
          << "Vertices (numVert):             " << numVert << "\n"
          << "Edges (numEdges):               " << numEdges << "\n"
          << "Max Allotment:                  " << max_allot << "\n"
          << "Temporary Storage:              " << temp_s << " bytes\n"
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

void GPU_BCG::init(cudaStream_t myStream) {
    Timer myTimer;

    // Initialize GPU memory
    long threadsPerBlock = 1024;
    long blocksPerGrid = (numEdges + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocksPerGrid, threadsPerBlock, 0, myStream>>>(
        d_level,            // 1. d_level array for bfs
        d_bcc_flag,         // 2. d_bcc_flag for updating bcc_numbers
        d_cut_vertex,       // 3. the vertex is a cut_vertex or not
        d_baseVertex,       // 4. every nonTree edge has an associated baseVertex(children of LCA)
        d_isSafe,           // 5. safeness associated with every vertex
        d_isPartofFund,     // 6. is the vertex part of any fundamental cycle or not
        d_is_baseVertex,    // 7. is the vertex a base vertex
        d_isFakeCutVertex,  // 8. to be used in the final iteration to identify all fake cut vertices 
        d_nonTreeEdgeId,    // 9. every vertex, if part of any fundamental cycle, needs to store the edge id
        numVert, numEdges); // 10. numVert and numEdges

    CUDA_CHECK(cudaPeekAtLastError(),           "Error during init_kernel launch"); // Check for any errors in kernel launch
    // Ensure all memset operations are completed before proceeding
    // CUDA_CHECK(cudaStreamSynchronize(myStream), "Failed to synchronize stream after init_kernel");

    auto dur = myTimer.stop();
    // std::cout <<"Initialization of device memory took : " << dur << " ms.\n";
}

GPU_BCG::~GPU_BCG() {
    // Free allocated memory
    Timer myTimer;
    if(g_verbose) {
        std::cout <<"\nDestructor started\n";
    }

    // Copy buffers
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaFree(d_edge_u[i]),       "Failed to free copy buffer_u");
        CUDA_CHECK(cudaFree(d_edge_v[i]),       "Failed to free copy buffer_v");
    }

    // Original_edge_stream
    CUDA_CHECK(cudaFree(original_u),            "Failed to free original_u array");
    CUDA_CHECK(cudaFree(original_v),            "Failed to free original_v array");

    // data - structures used for creating csr
    CUDA_CHECK(cudaFree(u_arr_buf),             "Failed to free u_arr array");
    CUDA_CHECK(cudaFree(v_arr_buf),             "Failed to free v_arr array");
    CUDA_CHECK(cudaFree(u_arr_alt_buf),         "Failed to free u_arr_alt_buf array");
    CUDA_CHECK(cudaFree(v_arr_alt_buf),         "Failed to free v_arr_alt_buf array");

    // csr vertex offset array
    CUDA_CHECK(cudaFree(d_vertices),            "Failed to free vertices array");

    // BFS output
    CUDA_CHECK(cudaFree(d_parent),              "Failed to free parent array");
    CUDA_CHECK(cudaFree(d_level),               "Failed to free level array");
    CUDA_CHECK(cudaFree(d_org_parent),          "Failed to free orginal_parent array");
    CUDA_CHECK(cudaFree(d_child_of_root),       "Failed to free d_child_of_root");

    // Part of cuda_BCC data-structure
    CUDA_CHECK(cudaFree(d_isSafe),              "Failed to free d_isSafe");
    CUDA_CHECK(cudaFree(d_baseVertex),          "Failed to free d_baseVertex");
    CUDA_CHECK(cudaFree(d_cut_vertex),          "Failed to free d_cut_vertex");
    CUDA_CHECK(cudaFree(d_imp_bcc_num),         "Failed to free d_imp_bcc_num");
    CUDA_CHECK(cudaFree(d_isPartofFund),        "Failed to free d_isPartofFund");
    CUDA_CHECK(cudaFree(d_is_baseVertex),       "Failed to free d_is_baseVertex");
    CUDA_CHECK(cudaFree(d_nonTreeEdgeId),       "Failed to free d_nonTreeEdgeId");

    // CC
    CUDA_CHECK(cudaFree(d_rep),                 "Failed to free d_rep");
    CUDA_CHECK(cudaFree(d_baseU),               "Failed to free d_baseU");
    CUDA_CHECK(cudaFree(d_baseV),               "Failed to free d_baseV");

    // For updating bcc_num
    CUDA_CHECK(cudaFree(d_bcc_flag),            "Failed to free d_bcc_flag");

    // BCG DS
    CUDA_CHECK(cudaFree(d_mapping),             "Failed to free d_mapping");
    CUDA_CHECK(cudaFree(d_isFakeCutVertex),     "Failed to free d_isFakeCutVertex");

    // Common flags
    CUDA_CHECK(cudaFree(d_flag),                "Failed to free d_flag");
    CUDA_CHECK(cudaFree(d_flags),               "Failed to free d_flags");
    CUDA_CHECK(cudaFree(d_temp_storage),        "Failed to free d_temp_storage");
    CUDA_CHECK(cudaFree(d_num_selected_out),    "Failed to free d_num_selected_out");
    
    // Destroy CUDA streams
    CUDA_CHECK(cudaStreamDestroy(computeStream),          "Failed to free computeStream");
    CUDA_CHECK(cudaStreamDestroy(transH2DStream),      "Failed to free transH2DStream");
    CUDA_CHECK(cudaStreamDestroy(transD2HStream),      "Failed to free transD2HStream");

    CUDA_CHECK(cudaEventDestroy(event),         "Failed to destroy event");

    if(g_verbose) {
        auto dur = myTimer.stop();
        std::cout <<"Deallocation of device memory took : " << dur << " ms.\n";
        std::cout <<"Destructor ended\n";
    }

    std::cout <<"\n[Process completed]" << std::endl;
}
