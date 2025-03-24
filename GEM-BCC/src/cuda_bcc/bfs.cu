#include "cuda_bcc/bfs.cuh"

__global__ 
void changeVariable(int* variable) {
    *variable = 0;
}

__global__ 
void setParentLevelKernel(int* d_parent, int* d_level, int* d_child_of_root, int root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_parent[root] = root;
        d_level[root] = 0;
        *d_child_of_root = -1;
    }
}

__global__ 
void simpleBFS(int no_of_vertices, int level, int* d_parents, int* d_levels, long* d_offset, int* d_neighbour, int* d_flag, int root, int* d_child_of_root) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < no_of_vertices && d_levels[tid] == level) {
        int u = tid;
        for (long i = d_offset[u]; i < d_offset[u + 1]; i++) {
            int v = d_neighbour[i];
            if(d_levels[v] < 0) {
                d_levels[v] = level + 1;
                d_parents[v] = u;
                *d_flag = 1;

                // Use atomicCAS for atomic compare-and-swap
                if (u == root && level == 0) {
                    int expected = -1;
                    atomicCAS(d_child_of_root, expected, v);
                }
            }
        }
    }
}

void constructSpanningTree(
    // Input variables
    int no_of_vertices,
    long numEdges,     
    long* d_offset,    
    int* d_neighbours, 
    int* d_flag,       
    int* h_flag,                       // Flags (host, pinned)
    int root,          
    
    // Output variables
    int* d_level,      
    int* d_parent,     
    int* d_child_of_root,              // Root's any child (device)
    int* h_child_of_root,              // Root's any child (host, pinned)
    
    // Stream variables
    cudaStream_t computeStream,        // Computation stream
    cudaStream_t transD2HStream)       // Device-to-host transfer stream
{

    *h_flag = 1;
    
    setParentLevelKernel<<<1, 1, 0, computeStream>>>(d_parent, d_level, d_child_of_root, root);

    if(g_verbose)
        kernelPrintCSRUnweighted(d_offset, d_neighbours, no_of_vertices, computeStream);

    int level = 0;
    int totalThreads = static_cast<int>(maxThreadsPerBlock);
    int no_of_blocks = (no_of_vertices + totalThreads - 1) / totalThreads;

    while (*h_flag) {

        *h_flag = 0;
        changeVariable<<<1, 1, 0, computeStream>>>(d_flag);
        
        simpleBFS<<<no_of_blocks, totalThreads, 0, computeStream>>>(
                    no_of_vertices, 
                    level, 
                    d_parent, 
                    d_level, 
                    d_offset, 
                    d_neighbours, 
                    d_flag, 
                    root, 
                    d_child_of_root);

        // Ensure 'h_flag' is updated before checking its value
        CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize stream");  
        CUDA_CHECK(cudaMemcpyAsync(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, transD2HStream), "cannot copy h_flag from gpu");
        CUDA_CHECK(cudaStreamSynchronize(transD2HStream), "Failed to synchronize stream");              
        ++level;
    }
    std::cout <<"Spanning Tree construction over.\n";
    
    if(g_verbose) {
        std::cout << "parent array: " << std::endl;
        kernelPrintArray(d_parent, no_of_vertices, computeStream);
    }

    CUDA_CHECK(cudaMemcpyAsync(h_child_of_root, d_child_of_root, sizeof(int), cudaMemcpyDeviceToHost, transD2HStream), "cannot copy child_of_root from gpu");
}

// ====[ End of constructSpanningTree Code ]====