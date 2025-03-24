#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

#include <cuda_runtime.h>

__global__
void init_kernel(int *counter, bool *d_isFakeCutVertex, const int numVerts) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numVerts) {
		counter[i] = 0;
		d_isFakeCutVertex[i] = 0;
	}
}

void init(bool *d_isFakeCutVertex, int *d_counter, const int& numVerts, cudaStream_t myStream) {
	int threads 	= 	1024; 
    int numBlocks 	= 	(numVerts + threads - 1) / threads;

    // step 0: reset the counter and fake_cut_vertices arrays
    init_kernel<<<numBlocks, threads, 0, myStream>>>(
    	d_counter,
    	d_isFakeCutVertex,  
    	numVerts);
	CUDA_CHECK(cudaGetLastError(), "init_kernel Kernel launch failed");
}

__global__ 
void freq_cal(int *d_mapping, int *d_counter, const int orig_numVerts){

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(tid < orig_numVerts) {
        int ver = d_mapping[tid];

        if(d_counter[ver] < 2){
            atomicAdd(d_counter + ver, 1);
        }
    }
}

__global__
void update_fake_cut_vertices(
	bool *d_isFakeCutVertex, 
	int *real_cut_vertex, const int *freq, const int numVerts) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < numVerts) {
		if(real_cut_vertex[tid] && (freq[tid] > 1)) {
            d_isFakeCutVertex[tid] = true;
            real_cut_vertex[tid] = 0;
        }
	}
}

void correct_misidentified_cut_vertices(
	bool *d_isFakeCutVertex, int *d_real_cut_vertex, 
	int *d_counter, int *d_mapping, 
	int numVerts, int orig_numVerts, cudaStream_t myStream) {

	int numThreads 	= static_cast<int>(maxThreadsPerBlock);
	int numBlocks 	= (orig_numVerts + numThreads - 1) / numThreads;
	
	// freq calculate
	freq_cal<<<numBlocks, numThreads, 0, myStream>>>(
		d_mapping, 
		d_counter, 
		orig_numVerts);

	CUDA_CHECK(cudaGetLastError(), "freq_cal Kernel launch failed");

	numBlocks = (numVerts + numThreads - 1) / numThreads;
	
	// update the cut vertices
	update_fake_cut_vertices<<<numBlocks, numThreads, 0, myStream>>>(
		d_isFakeCutVertex, 
		d_real_cut_vertex,
		d_counter, 
		numVerts);

	CUDA_CHECK(cudaGetLastError(), "update_fake_cut_vertices Kernel launch failed");
}

__global__ 
void update_bcc_numbers_kernel(
	bool *isSafe, 
	int *isRealCutVertex, 
	bool *d_isFakeCutVertex, 
	int *parent, 
	int *representative, 
	int *imp_bcc_num, 
	int *baseVertex, 
	long *nonTreeEdgeId, 
	int root, int numVerts) {

    // try share memory
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < numVerts) {
        int u = tid;

        do {
            while(isSafe[u]) {
        		long i = nonTreeEdgeId[parent[u]];
            	int b = baseVertex[i];
            	u = representative[b];	
        	}
            while(d_isFakeCutVertex[parent[u]] && u != root) {
            	u = parent[u];
            }

            // The loop will continue if isSafe[u] is true, otherwise it will break.
        } while(isSafe[u]);

        // After breaking out of the loop
    	imp_bcc_num[tid] = representative[u];
    }
}

__global__ 
void update_root_bcc_num(int* d_imp_bcc_num, int root, int child_of_root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_imp_bcc_num[root] = d_imp_bcc_num[child_of_root];
    }
}

void update_bcc_numbers(
	bool *d_isSafe, bool *d_isFakeCutVertex,
	int *d_isRealCutVertex, int *d_parent, 
	int *d_rep, int *d_imp_bcc_num, 
	int *d_baseVertex, long *d_nonTreeEdgeId, 
	int root, int numVerts, 
	cudaStream_t myStream) {

    // Define block and grid sizes
    int numThreads 	= static_cast<int>(maxThreadsPerBlock);
    int numBlocks 	= (numVerts + numThreads - 1) / numThreads;

    // Call the kernel
    update_bcc_numbers_kernel<<<numBlocks, numThreads, 0, myStream>>>(
    	d_isSafe, 
    	d_isRealCutVertex, 
    	d_isFakeCutVertex, 
    	d_parent, 
    	d_rep, 
    	d_imp_bcc_num, 
    	d_baseVertex, 
    	d_nonTreeEdgeId, 
    	root, numVerts);

	CUDA_CHECK(cudaGetLastError(), "Find_Unsafe_Component Kernel launch failed");
}

__global__
void cut_vert_bcc_update_kernel(
	int* is_real_cut, int* d_imp_bcc_num, 
	int* d_cut_ps, const int numVerts) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numVerts) {
		if(is_real_cut[i])
			d_imp_bcc_num[i] = numVerts + d_cut_ps[i];
	}
}

void repair(
	GPU_BCG& g_bcg_ds, const int& root, 
	const int child_of_root, const int totalRootChild, 
	cudaStream_t myStream) {
	
	//------------------------- Inputs -------------------------------
	int numVerts			=  	g_bcg_ds.numVert;
	int orig_numVert 		=  	g_bcg_ds.orig_numVert;

	int *d_rep 				=  	g_bcg_ds.d_rep; 
	int *d_parent 			=  	g_bcg_ds.d_parent;
	int *d_counter			=	g_bcg_ds.u_arr_buf;
	int *d_mapping 			= 	g_bcg_ds.d_mapping; // Vertex ID mapping
	int *d_cutVertex 		=  	g_bcg_ds.d_cut_vertex;
	int *d_baseVertex 		=  	g_bcg_ds.d_baseVertex;
	int *d_imp_bcc_num 		=  	g_bcg_ds.d_imp_bcc_num;
	long *d_nonTreeEdgeId 	=  	g_bcg_ds.d_nonTreeEdgeId;

	bool *isSafe 			=  	g_bcg_ds.d_isSafe;
	bool *d_isFakeCutVertex =	g_bcg_ds.d_isFakeCutVertex;
	//-----------------------------------------------------------------

	if(g_verbose) {
		std::cout << "Inside repair.\n";
		std::cout << "CUT Vertices info: \n";
		kernelPrintArray(d_cutVertex, numVerts, g_bcg_ds.computeStream);
	}

	std::cout <<"executing repair function." << "\n";
	// step 0: init the default values
	init(d_isFakeCutVertex, d_counter, numVerts, myStream);
	// step 1: identify & update all fake cut vertices
	correct_misidentified_cut_vertices(
		d_isFakeCutVertex, 
		d_cutVertex,
		d_counter, 
		d_mapping,
		numVerts, 
		orig_numVert, 
		myStream);
	
	// step 2: update implicit bcc numbers
	update_bcc_numbers(
		isSafe, 				// safeness of each vertex
		d_isFakeCutVertex, 		// fake cut vertices array
		d_cutVertex, 			// real cut vertices array
		d_parent, 				// parent array
		d_rep, 					// rep of every vertex
		d_imp_bcc_num, 			// output array
		d_baseVertex, 			// base vertex associated with every nonTree_id
		d_nonTreeEdgeId, 		// nonTree_id of every vertex
		root, 					// root
		numVerts, 				// numVert
		myStream);

	// step 3: handle root node (update bcc num)
	if(totalRootChild == 1) {
		// Root is not a cut vertex
	    // Copy the BCC number from the child node to the root node	
		update_root_bcc_num<<<1, 1, 0, myStream>>>(d_imp_bcc_num, root, child_of_root);
    }
    // Final synchronization of the stream to ensure all operations are complete
    CUDA_CHECK(cudaStreamSynchronize(myStream), "Failed to synchronize myStream after all operations");
}

// ====[ End of repair Code ]====