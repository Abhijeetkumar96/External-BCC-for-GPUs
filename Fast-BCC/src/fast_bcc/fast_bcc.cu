#include <cuda_runtime.h>
#include <iostream>

#include "fast_bcc/fast_bcc.cuh"
#include "utility/cuda_utility.cuh"
#include "fast_bcc/spanning_tree.cuh"
#include "fast_bcc/sparse_table_min.cuh"
#include "fast_bcc/sparse_table_max.cuh"
#include "fast_bcc/connected_components.cuh"

#define tb_size 1024
#define ins_batches 10
#define LOCAL_BLOCK_SIZE 1000
// #define DEBUG

__global__
void init_arrays(int* iscutVertex, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		iscutVertex[tid] = 0;
	}
}

__global__
void give_label_to_root(int* rep, int* parent, int numVert, int root) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numVert) {
        int temp = parent[idx];
        if(idx != temp && temp == root ){
            rep[temp] = rep[idx];
        }
    }
}

__global__
void checking(int* rep, int* parent, int numVert, int* d_flag_for_root ,int root) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // if(idx == 0) {
    //     printf("root flag: %d\n", *d_flag_for_root);
    // }

    if (idx < numVert) {
        int temp = parent[idx];
        if(idx != temp && rep[idx] != rep[temp] && temp == root){
            *d_flag_for_root = 1;            
        }
    }
}

__global__
void mark_cut_vertices(
    int* rep , int* parent, 
    int numVert , 
    int* d_flag_for_root , 
    int root , int* iscutVertex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        int temp = parent[idx];
        if(temp == root) {
            if(d_flag_for_root[0]) { 
            iscutVertex[temp] = true;
            }
        }
        else{
            if(rep[temp] != rep[idx]){
                iscutVertex[temp] = true;
            }
        }
    }
}

__global__
void init_w1_w2(
    int* w1, int* w2, 
    int* temp_label, 
    uint64_t* sf_edges, 
    int numVert, 
    int* first_occ, int* last_occ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        w1[idx] = first_occ[idx];
        w2[idx] = last_occ[idx];
        temp_label[idx] = idx;
        sf_edges[idx] = INT_MAX;
    }
}

__device__ inline 
int find_compress_SF(int i, int* temp_label) {
    int j = i;
    if (temp_label[j] == j) {
        return j;
    }
    do {
        j = temp_label[j];
    } while (temp_label[j] != j);

    int tmp;
    while((tmp = temp_label[i]) > j) {
        temp_label[i] = j;
        i = tmp;
    }
    return j;
}

__device__ inline 
bool union_async_SF(
    long idx, int src, int dst, 
    int* temp_label, uint64_t* sf_edges, uint64_t* edge_list) {
    while(1) {
        int u = find_compress_SF(src, temp_label);
        int v = find_compress_SF(dst, temp_label);

        if(u == v) break;
        
        if(v > u) { int temp; temp = u; u = v; v = temp; }
        
        if(u == atomicCAS(&temp_label[u], u, v)) {
           sf_edges[u] = edge_list[idx];
           return true;
        } 
    }
    return false;
}

__global__ 
void fill_w1_w2(uint64_t* edge_list, long numEdges, 
    int* w1, int* w2, int* parent, int* temp_label,
    uint64_t* sf_edges, int* first_occ , int* last_occ) {

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < numEdges){
        int u = (edge_list[idx] >> 32) & 0xFFFFFFFF;
        int v = edge_list[idx] & 0xFFFFFFFF;

        int f_u = first_occ[u];
        int l_u = last_occ[u];
        int f_v = first_occ[v];
        int l_v = last_occ[v];

        // if it is a non-tree edge
        if(u < v and (parent[u] != v) and (parent[v] != u)) {
            // Checking if the edge is a back-edge; if yes then update the tages
            /*
             * A back edge connects a vertex to one of its ancestors:
             *   - For a back edge where u is a descendant of v, then:
             *         first_occ[v] < first_occ[u] && last_occ[u] < last_occ[v]
             *   - For a back edge where v is a descendant of u, then:
             *         first_occ[u] < first_occ[v] && last_occ[v] < last_occ[u]
             *
             * If neither condition holds, the edge is considered a cross edge.
             */
            if ((f_v < f_u and l_u < l_v) or (f_u < f_v && l_v < l_u)) {
                // It is a back edge
                // printf("u: %d, v: %d is a back-edge\n", u, v);
                if(f_u < f_v) {
                    atomicMin(&w1[v], f_u);
                    atomicMax(&w2[u], f_v);
                }
                else {
                    atomicMin(&w1[u], f_v);
                    atomicMax(&w2[v], f_u);
                }
            }
            else {
                // else construct the forest as it is a cross-edge
                // printf("u: %d, v: %d is a cross-edge\n", u, v);
                bool r = union_async_SF(idx, u, v, temp_label, sf_edges, edge_list);
            }
        }
    }
}

__global__
void update_w1_w2(
    uint64_t* sf_edges, int numVert, 
    int* w1, int* w2, 
    int* parent, int* first_occ, int* last_occ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if (idx < numVert) {
        if(sf_edges[idx] == INT_MAX) return;

        int u = (sf_edges[idx] >> 32) & 0xFFFFFFFF;
        int v = sf_edges[idx] & 0xFFFFFFFF;
        
        if(u < v) {
            if(first_occ[u] < first_occ[v]) {
                atomicMin(&w1[v], first_occ[u]);
                atomicMax(&w2[u], first_occ[v]);
            }
            else {
                atomicMin(&w1[u], first_occ[v]);
                atomicMax(&w2[v], first_occ[u]);
            }
        }
    }
}


__global__
void compute_a1(int* first_occ, int* last_occ, int numVert , int* w1 , int* a1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        a1[first_occ[idx]] = w1[idx];
        a1[last_occ[idx]] = w1[idx];
    }
}

__global__
void fill_left_right(int* first_occ , int* last_occ, int numVert, int* left, int* right) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        left[idx] = first_occ[idx];
        right[idx] = last_occ[idx];
    }
}

void assign_tags(const int root, GPU_BCG& g_bcg_ds, bool isLastBatch) {

	int numVert             = 	g_bcg_ds.numVert;
    int numEdges            = 	g_bcg_ds.numEdges;
    
    int* d_first_occ        =   g_bcg_ds.d_first_occ;
    int* d_last_occ         =   g_bcg_ds.d_last_occ;
    uint64_t* d_edgelist    =   g_bcg_ds.updated_edgelist;
    int* d_parent           =   g_bcg_ds.d_parent;
    
    int* d_w1               =   g_bcg_ds.d_w1;
    int* d_w2               =   g_bcg_ds.d_w2;

    int* d_left             =   g_bcg_ds.d_left;
    int* d_right            =   g_bcg_ds.d_right;

    int* d_a1               =   g_bcg_ds.d_a1;
    int* d_a2               =   g_bcg_ds.d_a2;
    
    int* d_na1              =   g_bcg_ds.d_na1;
    int* d_na2              =   g_bcg_ds.d_na2;
    
    int* d_low              =   g_bcg_ds.d_low;
    int* d_high             =   g_bcg_ds.d_high;
    
    int* temp_label         =   g_bcg_ds.d_fg;

    uint64_t* sf_edges      =   g_bcg_ds.d_index; // of size numVert

    int n_asize = (2 * numVert + LOCAL_BLOCK_SIZE - 1) / LOCAL_BLOCK_SIZE;

   // step 2: Compute w1, w2, low and high using first and last
    init_w1_w2<<<CEIL(numVert, tb_size), tb_size>>>(
        d_w1, 
        d_w2, 
        temp_label,
        sf_edges,
        numVert, 
        d_first_occ, 
        d_last_occ);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    fill_w1_w2<<<CEIL(numEdges, tb_size), tb_size>>>(
        d_edgelist, 
        numEdges, 
        d_w1, 
        d_w2, 
        d_parent,
        temp_label,
        sf_edges, 
        d_first_occ, 
        d_last_occ);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    update_w1_w2<<<CEIL(numVert, tb_size), tb_size>>>(
        sf_edges, 
        numVert, 
        d_w1, 
        d_w2, 
        d_parent, 
        d_first_occ, 
        d_last_occ
    );
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    compute_a1<<<CEIL(numVert, tb_size), tb_size>>>(
        d_first_occ, d_last_occ, 
        numVert, 
        d_w1,
        d_a1
    );
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    compute_a1<<<CEIL(numVert, tb_size), tb_size>>>(
        d_first_occ, 
        d_last_occ, 
        numVert, 
        d_w2, 
        d_a2
    );
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    fill_left_right<<<CEIL(numEdges, tb_size), tb_size>>>(
        d_first_occ, 
        d_last_occ, 
        numVert, 
        d_left, 
        d_right
    );
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    main_min(2*numVert, numVert, d_a1, d_left, d_right, d_low, n_asize , d_na1);
    main_max(2*numVert, numVert, d_a2, d_left, d_right, d_high, n_asize , d_na2);
}

void finalise_labels(int root, GPU_BCG& g_bcg_ds) {

	int* d_rep 		= 	g_bcg_ds.d_rep;
	int* d_parent 	= 	g_bcg_ds.d_parent;
	int numVert 	= 	g_bcg_ds.numVert;
	int* iscutVertex =  g_bcg_ds.iscutVertex;


    const long numThreads = 1024;
    int numBlocks = (numVert + numThreads - 1) / numThreads;

	init_arrays<<<numBlocks, numThreads>>>(iscutVertex, numVert);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

	give_label_to_root<<<(numVert + 1023)/ 1024 , 1024>>>(
        d_rep, d_parent, 
        numVert, 
        root);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    int* d_flag_for_root = g_bcg_ds.d_flag;
    int h_flag_for_root = 0;
    CUDA_CHECK(cudaMemcpy(d_flag_for_root, &h_flag_for_root, sizeof(int), cudaMemcpyHostToDevice), "Failed to copy");

    checking<<<(numVert+ 1023 )/ 1024 , 1024>>>(d_rep, d_parent, numVert, d_flag_for_root, root);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    #ifdef DEBUG
        std::cout << "Flag array after checking kernel: ";
        print_device_array(d_flag_for_root , 1);

        std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        std::cout << "BCC Numbers:" << "\n";
        print_device_array(d_rep, numVert);
        std::cout << "\n";
        std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    #endif

    mark_cut_vertices<<<(numVert + 1023) / 1024, 1024>>>(
        d_rep, d_parent, 
        numVert, 
        d_flag_for_root, 
        root, 
        iscutVertex);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    #ifdef DEBUG
        std::cout << "Cut Vertices info: \n" << "\n";
        print_device_array(iscutVertex, numVert);
    #endif
}

__global__ 
void update_bcc_flag_kernel(int* d_cut_vertex, int* d_bcc_num, int* d_bcc_flag, int numVert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numVert) {
        if(!d_cut_vertex[i]) {
            d_bcc_flag[d_bcc_num[i]] = 1;
        }
    }
}

__global__ 
void update_bcc_number_kernel(int* d_cut_vertex, int* d_bcc_num, int* bcc_ps, int* cut_ps, int numVert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if(i == 0) update_bcg_num_vert;
    if(i < numVert) {
        if(!d_cut_vertex[i]) {
            d_bcc_num[i] = bcc_ps[d_bcc_num[i]] - 1;
        }
        else
            d_bcc_num[i] = bcc_ps[numVert - 1] + cut_ps[i] - 1;
    }
}

// inclusive prefix_sum
void incl_scan(
    int*& d_in, 
    int*& d_out, 
    int& num_items) {

    size_t temp_storage_bytes = 0;
    cudaError_t status;

    status = cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, d_in, d_out, num_items);
    CUDA_CHECK(status, "Error in CUB InclusiveSum");

    // Allocate once (e.g., during initialization)
    void* d_temp_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate temporary storage for CUB");
    
    // Run inclusive prefix sum
    status = cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    CUDA_CHECK(status, "Error in CUB InclusiveSum");

    CUDA_CHECK(cudaFree(d_temp_storage), "Failed to free");
}

int update_bcc_numbers(GPU_BCG& g_bcg_ds, int numVert) {
    int* d_bcc_flag     =   g_bcg_ds.d_bcc_flag;
    int* d_bcc_ps       =   g_bcg_ds.d_left;   // reusing few arrays
    int* d_cut_ps       =   g_bcg_ds.d_right; // reusing few arrays
    
    int* d_cut_vertex   =   g_bcg_ds.iscutVertex;
    int* d_bcc_num      =   g_bcg_ds.d_rep;
    
    int numThreads = static_cast<int>(maxThreadsPerBlock);
    size_t numBlocks = (numVert + numThreads - 1) / numThreads;

    init_arrays<<<numBlocks, numThreads>>>(d_bcc_flag, numVert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init_arrays kernel");

    update_bcc_flag_kernel<<<numBlocks, numThreads>>>(d_cut_vertex, d_bcc_num, d_bcc_flag, numVert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_bcc_flag kernel");

    incl_scan(d_bcc_flag,   d_bcc_ps, numVert);
    incl_scan(d_cut_vertex, d_cut_ps, numVert);

    // pinned memory
    int* h_max_ps_bcc           = g_bcg_ds.h_max_ps_bcc;
    int* h_max_ps_cut_vertex    = g_bcg_ds.h_max_ps_cut_vertex;

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");
    CUDA_CHECK(cudaMemcpy(h_max_ps_bcc, &d_bcc_ps[numVert - 1], sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back max_ps_bcc.");
    CUDA_CHECK(cudaMemcpy(h_max_ps_cut_vertex, &d_cut_ps[numVert - 1], sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back max_ps_cut_vertex.");
    
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");

    // std::cout << "max_ps_bcc: " << *h_max_ps_bcc << "\n";
    // std::cout << "max_ps_cut_vertex: " << *h_max_ps_cut_vertex << "\n" << std::endl;

    int bcg_num_vert = *h_max_ps_bcc + *h_max_ps_cut_vertex;

    update_bcc_number_kernel<<<numBlocks, numThreads>>>(
        d_cut_vertex, 
        d_bcc_num, 
        d_bcc_ps, 
        d_cut_ps, 
        numVert
    );

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");

    if(g_verbose) {
        std::cout << "BCC numbers:" << "\n";
        print_device_array(d_bcc_num, numVert);
        std::cout << "Cut vertex status:" << "\n";
        print_device_array(d_cut_vertex, numVert);
    }

    return bcg_num_vert;
}

void Fast_BCC(GPU_BCG& g_bcg_ds, const bool isLastBatch) {

    int numVert     =   g_bcg_ds.numVert;
    long numEdges    =   g_bcg_ds.numEdges;

    if(g_verbose) {
        std::cout << "\n***************** Starting CUDA_BCC...*****************\n";
        std::cout << "Input numVert: " << numVert << " and numEdges: " << numEdges << "\n";
        std::cout << "********************************************************************\n" << std::endl;
    }

    // step 0: init cut vertices, cut edges, bcc numbers, d_fg, etc

	// step 1: construct Spanning Tree
	int root = construct_spanning_tree(g_bcg_ds);
    g_bcg_ds.last_root = root;

    // std::cout << "Root: " << root << std::endl;

    if(isLastBatch and g_verbose) {
        std::cout << "Actual Edges array:" << "\n";
        print_device_edges(g_bcg_ds.updated_edgelist, numEdges);
        std::cout << "\n";
    }

    auto start = std::chrono::high_resolution_clock::now();
	// step 2: Assigning Tags
	assign_tags(root, g_bcg_ds, isLastBatch);

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "One round of Assigning Tags takes: " << dur << " ms." << "\n" << std::endl;

    start = std::chrono::high_resolution_clock::now();
	// step 3: Apply CC

    // update this code
	CC(g_bcg_ds, isLastBatch);

    // end = std::chrono::high_resolution_clock::now();
    // dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "One round of Last CC takes: " << dur << " ms." << "\n" << std::endl;

    if(isLastBatch)
        return;

	start = std::chrono::high_resolution_clock::now();
    // step 4: finalise the labels
	finalise_labels(root, g_bcg_ds);

    // end = std::chrono::high_resolution_clock::now();
    // dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "One round of Finalising Labels takes: " << dur << " ms." << "\n";

    start = std::chrono::high_resolution_clock::now();
    // step 5: update BCC numbers
    g_bcg_ds.numVert = update_bcc_numbers(g_bcg_ds, numVert);

    // end = std::chrono::high_resolution_clock::now();
    // dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "One round of Updating BCC numbers takes: " << dur << " ms." << "\n" << std::endl;
}