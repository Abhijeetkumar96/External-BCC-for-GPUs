#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// Helper functions
#include "utility/timer.hpp"
#include "utility/utility.hpp"
#include "utility/cuda_utility.cuh"

#include "cuda_bcc/bcc.cuh"

#include "cuda_bcc/cuda_csr.cuh"
#include "cuda_bcc/bfs.cuh"
#include "cuda_bcc/lca.cuh"
#include "cuda_bcc/cut_vertex.cuh"

#include "extern_bcc/bcg_memory_utils.cuh"

__global__ 
void update_bcc_flag_kernel(int* d_cut_vertex, int* d_bcc_num, int* d_flag, int numVert) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numVert) {
		if(!d_cut_vertex[i]) {
			d_flag[d_bcc_num[i]] = 1;
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
	int& num_items, 
	void*& d_temp_storage, 
	cudaStream_t myStream) {

    size_t temp_storage_bytes = 0;
    cudaError_t status;

    status = cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, d_in, d_out, num_items, myStream);
    CUDA_CHECK(status, "Error in CUB InclusiveSum");
    
    // Run inclusive prefix sum
    status = cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, myStream);
    CUDA_CHECK(status, "Error in CUB InclusiveSum");
}

__global__
void label_bcc_art(int numVert, int *d_parent, int *d_cut_vertex, int *d_imp_bcc_num) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < numVert) {
		d_cut_vertex[d_parent[tid]] = 1;
		d_imp_bcc_num[tid] = tid;
	}
}

void label_tree_bcc_art(GPU_BCG& g_bcg_ds, cudaStream_t myStream) {

	int numThreads 	= static_cast<int>(maxThreadsPerBlock);
	int numBlocks 	= (g_bcg_ds.numVert + numThreads - 1) / numThreads;

	// all the leaf nodes are non cut - vertices, remaining all vertices are cut-vertices.
	label_bcc_art<<<numBlocks, numThreads, 0, myStream>>>(
		g_bcg_ds.numVert,			// input size
		g_bcg_ds.d_parent, 			// input
		g_bcg_ds.d_cut_vertex, 		// output
		g_bcg_ds.d_imp_bcc_num	 	// output
		);

	CUDA_CHECK(cudaGetLastError(), "updateCutVertex Kernel launch failed");
}


int update_bcc_numbers(GPU_BCG& g_bcg_ds, int numVert) {
    int* d_bcc_flag 	= 	g_bcg_ds.d_bcc_flag;
    int* d_bcc_ps 		= 	g_bcg_ds.d_rep;   // reusing few arrays
    int* d_cut_ps 		= 	g_bcg_ds.d_level; // reusing few arrays
    
    int* d_cut_vertex 	= 	g_bcg_ds.d_cut_vertex;
    int* d_bcc_num 		= 	g_bcg_ds.d_imp_bcc_num;

    cudaStream_t computeStream 	= g_bcg_ds.computeStream;
    cudaStream_t transD2HStream = g_bcg_ds.transD2HStream;
    
    int numThreads = static_cast<int>(maxThreadsPerBlock);
    size_t numBlocks = (numVert + numThreads - 1) / numThreads;
    
    update_bcc_flag_kernel<<<numBlocks, numThreads, 0, computeStream>>>(d_cut_vertex, d_bcc_num, d_bcc_flag, numVert);
    CUDA_CHECK(cudaStreamSynchronize(computeStream), "..");

    incl_scan(d_bcc_flag,   d_bcc_ps, numVert, g_bcg_ds.d_temp_storage, computeStream);
    incl_scan(d_cut_vertex, d_cut_ps, numVert, g_bcg_ds.d_temp_storage, computeStream);

    // pinned memory
    int* h_max_ps_bcc		 	= g_bcg_ds.h_max_ps_bcc;
    int* h_max_ps_cut_vertex 	= g_bcg_ds.h_max_ps_cut_vertex;

    CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize stream after copying max prefix sums");

    CUDA_CHECK(cudaMemcpyAsync(h_max_ps_bcc, &d_bcc_ps[numVert - 1], sizeof(int), cudaMemcpyDeviceToHost, transD2HStream), "Failed to copy back max_ps_bcc.");

    CUDA_CHECK(cudaMemcpyAsync(h_max_ps_cut_vertex, &d_cut_ps[numVert - 1], sizeof(int), cudaMemcpyDeviceToHost, transD2HStream), "Failed to copy back max_ps_cut_vertex.");
    
    CUDA_CHECK(cudaStreamSynchronize(transD2HStream), "Failed to synchronize stream after copying max prefix sums");

	std::cout << "max_ps_bcc: " << *h_max_ps_bcc << "\n";
	std::cout << "max_ps_cut_vertex: " << *h_max_ps_cut_vertex << "\n";

    int bcg_num_vert = *h_max_ps_bcc + *h_max_ps_cut_vertex;

    update_bcc_number_kernel<<<numBlocks, numThreads, 0, computeStream>>>(
    	d_cut_vertex, 
    	d_bcc_num, 
    	d_bcc_ps, 
    	d_cut_ps, 
    	numVert
    );

    CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize stream after copying max prefix sums");

    if(g_verbose) {
    	std::cout << "BCC numbers:" << std::endl;
    	kernelPrintArray(d_bcc_num, numVert, computeStream);
    	std::cout << "Cut vertex status:" << std::endl;
    	kernelPrintArray(d_cut_vertex, numVert, computeStream);
    }

    return bcg_num_vert;
}

void cuda_bcc(GPU_BCG& g_bcg_ds, bool isLastBatch) {

	int *original_u 		= 	g_bcg_ds.original_u;
	int *original_v 		= 	g_bcg_ds.original_v;

	void *cub_temp_storage 	=  	g_bcg_ds.d_temp_storage;

	// numEdges is unique edge count (only (2,1), not (1,2)).
	long numEdges 			= 	g_bcg_ds.numEdges;
	int numVert 			=   g_bcg_ds.numVert;
	bool isTree  			=   false;

	std::cout << "inside cuda_bcc " << std::endl;
	std::cout << "numVert = " 	<< numVert << "\n";
	std::cout << "numEdges = " 	<< numEdges << "\n";
	std::cout << "isLastBatch = " << isLastBatch << "\n";
	std::cout << std::endl;

	// Check if the input graph is a tree.
	if (numEdges == numVert - 1) {
		std::cout <<"input graph is a tree." << "\n";
		isTree = true;
	}

	if(numEdges < numVert - 2) {
		std::cerr <<"numEdges less than numVert. Exiting.\n";
		exit(0);
	}

	g_bcg_ds.init(g_bcg_ds.computeStream);

	// step 1: Create duplicates
	int *u_arr_buf = g_bcg_ds.u_arr_buf;
	int *v_arr_buf = g_bcg_ds.v_arr_buf;

	long E = 2 * numEdges; // Two times the original edges count (0,1) and (1,0).

	create_duplicate(original_u, original_v, u_arr_buf, v_arr_buf, numEdges, g_bcg_ds.computeStream);

	// Step [i]: alternate buffers for sorting operation
	int *u_arr_alt_buf = g_bcg_ds.u_arr_alt_buf;
	int *v_arr_alt_buf = g_bcg_ds.v_arr_alt_buf;

	// Create DoubleBuffers
    cub::DoubleBuffer<int> d_u_arr(u_arr_buf, u_arr_alt_buf);
    cub::DoubleBuffer<int> d_v_arr(v_arr_buf, v_arr_alt_buf);

    // Step [ii]: Output buffer for csr
    long* d_vertices = g_bcg_ds.d_vertices;

    // Output: 
    // Vertices array			-> d_vertices <- type: long;
    // Neighbour/edges array	-> d_v_arr.Current() <- type: int;

    Timer myTimer;
    gpu_csr(d_u_arr, d_v_arr, cub_temp_storage, E, numVert, d_vertices, g_bcg_ds.computeStream);

	// CSR creation ends here

	int *d_parent = g_bcg_ds.d_parent;
	int *d_level = g_bcg_ds.d_level;

	// Create a random device and seed it
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a distribution in the range [0, numVert]
    std::uniform_int_distribution<> distrib(0, numVert - 1);

    // Generate a random root value
    int root = distrib(gen);
    // root = 11;
    // Output the random root value
    std::cout << "Random root value: " << root << "\n";

	// Step 1: Construct a rooted spanning tree
	constructSpanningTree(
	    numVert,                          // Number of vertices
	    E,                                // Number of edges
	    d_vertices,                       // Device offset array
	    d_v_arr.Current(),                // Device neighbors array
	    g_bcg_ds.d_flag,                  // Device flag array
	    g_bcg_ds.h_flag,                  // Host flag array (pinned memory)
	    root,                             // Root vertex for spanning tree

	    d_level,                          // Output: Device array of vertex levels
	    d_parent,                         // Output: Device array of vertex parents
	    g_bcg_ds.d_child_of_root,         // Output: Device count of root's children
	    g_bcg_ds.h_child_of_root,         // Output: Host count of root's children (pinned memory)
	    g_bcg_ds.computeStream,           // CUDA compute stream
	    g_bcg_ds.transD2HStream           // CUDA Device-to-Host transfer stream
	);

	int* h_child_of_root = g_bcg_ds.h_child_of_root;

	if(isTree) {
		label_tree_bcc_art(g_bcg_ds, g_bcg_ds.computeStream);
		g_bcg_ds.numVert = update_bcc_numbers(g_bcg_ds, numVert);

		return;
	}

	// Step 3 & 4 : Find LCA and Base Vertices, then apply connected Comp
    naive_lca(g_bcg_ds, root);

    // Step 5: Propagate safness to representative & parents
    // Step 6: Update cut vertex status and cut - edge status
    // Step 7: Update implicit bcc labels
    assign_cut_vertex_BCC(g_bcg_ds, root, *h_child_of_root, false, isLastBatch);

	// update_bcc_numbers, if not the last batch
	if(!isLastBatch)
		g_bcg_ds.numVert = update_bcc_numbers(g_bcg_ds, numVert);

}

// ====[ End of cuda_bcc Code ]====