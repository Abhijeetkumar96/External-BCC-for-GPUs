#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

#include <cuda_runtime.h>

// #define DEBUG

__global__
void init_kernel(int *counter, int *d_isFakeCutVertex, const int numVerts) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numVerts) {
		counter[i] = 0;
		d_isFakeCutVertex[i] = 0;
	}
}

void init(int *d_isFakeCutVertex, int *d_counter, const int& numVerts) {
	int threads 	= 	1024; 
    int numBlocks 	= 	(numVerts + threads - 1) / threads;

    // step 0: reset the counter and fake_cut_vertices arrays
    init_kernel<<<numBlocks, threads>>>(
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
	int *d_isFakeCutVertex, 
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
	int *d_isFakeCutVertex, int *d_real_cut_vertex, 
	int *d_counter, int *d_mapping, 
	int numVerts, int orig_numVerts) {

	int numThreads 	= static_cast<int>(maxThreadsPerBlock);
	int numBlocks 	= (orig_numVerts + numThreads - 1) / numThreads;
	
	// freq calculate
	freq_cal<<<numBlocks, numThreads>>>(
		d_mapping, 
		d_counter, 
		orig_numVerts);

	CUDA_CHECK(cudaGetLastError(), "freq_cal Kernel launch failed");
	CUDA_CHECK(cudaDeviceSynchronize(), "freq_cal Kernel launch failed");

	numBlocks = (numVerts + numThreads - 1) / numThreads;
	
	// update the cut vertices
	// update_fake_cut_vertices<<<numBlocks, numThreads>>>(
	// 	d_isFakeCutVertex, 
	// 	d_real_cut_vertex,
	// 	d_counter, 
	// 	numVerts);

	// CUDA_CHECK(cudaDeviceSynchronize(), "update_fake_cut_vertices Kernel launch failed");

	#ifdef DEBUG
		std::cout << "Printing from Repair function" << std::endl;
		std::cout << "d_counter array:" << std::endl;
		DisplayDeviceArray(d_counter, numVerts);

		std::cout << "d_isFakeCutVertex array:" << std::endl;
		DisplayDeviceArray(d_isFakeCutVertex, numVerts);
	#endif
}

void repair(GPU_BCG& g_bcg_ds) {
	
	//------------------------- Inputs -------------------------------
	int numVerts			=  	g_bcg_ds.numVert;
	int orig_numVert 		=  	g_bcg_ds.orig_numVert;

	int *d_counter			=	g_bcg_ds.d_counter;
	int *d_mapping 			= 	g_bcg_ds.d_mapping; // Vertex ID mapping
	int *d_cutVertex 		=  	g_bcg_ds.iscutVertex;
	int *d_isFakeCutVertex =	g_bcg_ds.d_isFakeCutVertex;
	//-----------------------------------------------------------------

	if(g_verbose) {
		std::cout << "Inside repair.\n";
		std::cout << "CUT Vertices info: \n";
		print_device_array(d_cutVertex, numVerts);
	}

	// std::cout <<"executing repair function." << "\n";
	// step 0: init the default values
	init(d_isFakeCutVertex, d_counter, numVerts);
	// step 1: identify & update all fake cut vertices
	correct_misidentified_cut_vertices(
		d_isFakeCutVertex, 
		d_cutVertex,
		d_counter, 
		d_mapping,
		numVerts, 
		orig_numVert);
    // Final synchronization of the stream to ensure all operations are complete
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize myStream after all operations");
}

// ====[ End of repair Code ]====