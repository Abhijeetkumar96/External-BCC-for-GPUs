#include "cuda_bcc/connected_components.cuh"
#include "utility/cuda_utility.cuh"

__global__
void initialise(int* parent, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		parent[tid] = tid;
	}
}

__global__ 
void change_variable(int* variable) {
    *variable = 0;
}

__global__ 
void hooking(
	long numEdges, 
	int* d_source, 
	int* d_destination, 
	int* d_rep, 
	int* d_flag, 
	int itr_no) {
	long tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid < numEdges) {
		int edge_u = d_source[tid];
		int edge_v = d_destination[tid];

		int comp_u = d_rep[edge_u];
		int comp_v = d_rep[edge_v];

		if(comp_u != comp_v) 
		{
			*d_flag = 1;
			int max = (comp_u > comp_v) ? comp_u : comp_v;
			int min = (comp_u < comp_v) ? comp_u : comp_v;

			if(itr_no%2) {
				d_rep[min] = max;
			}
			else { 
				d_rep[max] = min;
			}
		}
	}
}

__global__ 
void short_cutting(int n, int* d_parent) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		if(d_parent[tid] != tid) {
			d_parent[tid] = d_parent[d_parent[tid]];
		}
	}	
}

void connected_comp(
	long numEdges, 
	int* u_arr, 
	int* v_arr, 
	int numVert, 
	int* d_rep, 
	int* h_flag,  		// pinned memory
	int* d_flag, 
	cudaStream_t computeStream,
	cudaStream_t transD2HStream) {

    const long numThreads = maxThreadsPerBlock;
    int numBlocks = (numVert + numThreads - 1) / numThreads;

    initialise<<<numBlocks, numThreads, 0, computeStream>>>(d_rep, numVert);
    cudaError_t err = cudaGetLastError();
    CUDA_CHECK(err, "Error in launching initialise kernel");

    *h_flag = 1;
    int iteration = 0;

    const long numBlocks_hooking = (numEdges + numThreads - 1) / numThreads;
    const long numBlocks_updating_parent = (numVert + numThreads - 1) / numThreads;

    while(*h_flag) {
        *h_flag = 0;
        iteration++;

        change_variable<<<1, 1, 0, computeStream>>>(d_flag);
        CUDA_CHECK(cudaStreamSynchronize(computeStream), "..");

        hooking<<<numBlocks_hooking, numThreads, 0, computeStream>>>(
        	numEdges, 
        	u_arr, 
        	v_arr, 
        	d_rep, 
        	d_flag, 
        	iteration
        	);
        err = cudaGetLastError();
        CUDA_CHECK(err, "Error in launching hooking kernel");

        // make sure that hooking is complete before copying flag back to host
        CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize computeStream");
        
        CUDA_CHECK(cudaMemcpyAsync(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, transD2HStream), 
        	"Unable to copy back h_flag to host");

        for(int i = 0; i < std::ceil(std::log2(numVert)); ++i) {
            short_cutting<<<numBlocks_updating_parent, numThreads, 0, computeStream>>>(numVert, d_rep);
            err = cudaGetLastError();
            CUDA_CHECK(err, "Error in launching short_cutting kernel");
        }
        
        CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize computeStream");
        CUDA_CHECK(cudaStreamSynchronize(transD2HStream), "Failed to synchronize transD2HStream");
        
    }
}

