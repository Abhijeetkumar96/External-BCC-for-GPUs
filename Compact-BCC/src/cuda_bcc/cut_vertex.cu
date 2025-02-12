//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "utility/utility.hpp"
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

//---------------------------------------------------------------------
// Cut vertices Specific Utilities
//---------------------------------------------------------------------
#include "cuda_bcc/cut_vertex.cuh"
#include "extern_bcc/repair.cuh"

//---------------------------------------------------------------------
// CUDA Kernels
//---------------------------------------------------------------------

__global__ 
void changevariable(int* variable) {
    *variable = 0;
}

__global__ 
void set_root_cut_status(int* d_cutVertex, int root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cutVertex[root] = 1;
    }
}

__global__ 
void update_root_bcc_number(int* d_imp_bcc_num, int root, int child_of_root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_imp_bcc_num[root] = d_imp_bcc_num[child_of_root];
    }
}

__global__ 
void Propagate_Safeness_to_rep(int totalVertices, bool *d_isBaseVertex, int *d_rep, bool *d_isSafe) {
	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < totalVertices) {
		if(d_isBaseVertex[tid] && d_isSafe[tid]) {
			d_isSafe[d_rep[tid]] = 1;
		}
	}
}

__global__ 
void Propagate_Safeness_to_comp(int totalVertices, int *d_rep, bool *d_isSafe) {
	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < totalVertices) {
		if(d_isSafe[d_rep[tid]]) {
			d_isSafe[tid] = 1;
		}
	}
}

__global__ 
void Find_Unsafe_Component(
	int root, 
	int totalVertices, 
	int *d_rep, 
	bool *d_isSafe, 
	int *d_cutVertex, 
	int *d_parent, 
	int *d_totalChild) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
  	
  	if(tid < totalVertices) {
		if(d_rep[tid] == tid && !d_isSafe[tid] && tid != root) {
			d_cutVertex[d_parent[tid]] = 1;

			// Optimize here
			if(d_parent[tid] == root) {
				d_cutVertex[root] = 0;
				atomicAdd(d_totalChild, 1);
			}
		}
	}
}

__global__ 
void updateCutVertex(
	int 	totalVertices, 
	int 	*d_parent,
	bool 	*d_partOfFundamental, 
	long 	*d_offset, 
	int 	*d_cutVertex) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

  	if(tid < totalVertices) {
  		int u = tid;
  		int v = d_parent[u];

		if (u == v) {
			return;
		}

		if (d_partOfFundamental[u]) {
			return;
		}
		
		if(d_offset[u+1] - d_offset[u] > 1) {
			d_cutVertex[u] = 1;
		}

		if(d_offset[v+1] - d_offset[v] > 1) {
			d_cutVertex[v] = 1;
		}
	}
}

__global__ 
void updateCutVertex_firstBatch(
	const int totalVertices, 
	const int *d_parent, 
	const bool *d_partOfFundamental, 
	const bool *is_parent, int *d_cutVertex) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

  	if(tid < totalVertices) {
  		int u = tid;
  		int v = d_parent[u];

		if (u == v) {
			return;
		}

		if (d_partOfFundamental[u]) {
			return;
		}
		
		// if u is not a leaf node, then update its cut_vertex status
		if(is_parent[u]) {
			d_cutVertex[u] = 1;
		}

		if(is_parent[v]) {
			d_cutVertex[v] = 1;
		}
	}
}

__global__
void updateLeafNode(int totalVertices, int *d_parent, bool *flag) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < totalVertices) {
		flag[d_parent[tid]] = 1;
	}
}

__global__ 
void implicit_bcc(
	int totalVertices, 
	bool *isSafe, 
	int *representative, 
	int *parent, 
	int *baseVertex, 
	long *nonTreeEdgeId, 
	int *unsafeRep) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < totalVertices) {
        int u = tid;

        while(isSafe[u]) {
            long i = nonTreeEdgeId[parent[u]];
            int b = baseVertex[i];
            u = representative[b];
        }

        unsafeRep[tid] = representative[u];
    }
}

void assign_cut_vertex_BCC(GPU_BCG& g_bcg_ds, int root, int child_of_root, bool isFirstBatch, bool isLastBatch) {

	//------------------------- Inputs -------------------------------
    int no_of_vertices          =    g_bcg_ds.numVert;
    int *d_totalRootChild       =    g_bcg_ds.d_flag;
    int *h_totalRootChild       =    g_bcg_ds.h_flag;

    int *d_parent               =    g_bcg_ds.d_parent;
    int *d_rep                  =    g_bcg_ds.d_rep;
    int *d_baseVertex           =    g_bcg_ds.d_baseVertex;

    bool *d_isSafe              =    g_bcg_ds.d_isSafe;
    bool *d_isBase              =    g_bcg_ds.d_is_baseVertex;
    bool *d_partOfFundamental   =    g_bcg_ds.d_isPartofFund;

    long *d_nonTreeEdgeId       =    g_bcg_ds.d_nonTreeEdgeId;
    long *d_offset              =    g_bcg_ds.d_vertices;

    //-----------------------------------------------------------------

    //------------------------- Outputs -------------------------------
    int *d_cutVertex            =    g_bcg_ds.d_cut_vertex;
    int *d_imp_bcc_num          =    g_bcg_ds.d_imp_bcc_num;
    //-----------------------------------------------------------------

    //------------------------- cuda streams -------------------------------
    // -> streams for computation and transfer
    cudaStream_t computeStream 	= g_bcg_ds.computeStream;
    cudaStream_t transD2HStream = g_bcg_ds.transD2HStream; 
    //-----------------------------------------------------------------
    
	int totalThreads = static_cast<int>(maxThreadsPerBlock);
	int no_of_blocks = (no_of_vertices + totalThreads - 1) / totalThreads;

    Propagate_Safeness_to_rep<<<no_of_blocks, totalThreads, 0, computeStream>>>(
    	no_of_vertices, 
    	d_isBase, 
    	d_rep, 
    	d_isSafe);

    CUDA_CHECK(cudaGetLastError(), "Propagate_Safeness_to_rep Kernel launch failed");
    
    Propagate_Safeness_to_comp<<<no_of_blocks, totalThreads, 0, computeStream>>>(
    	no_of_vertices, 
    	d_rep, 
    	d_isSafe);

    CUDA_CHECK(cudaGetLastError(), "Propagate_Safeness_to_comp Kernel launch failed");

    changevariable<<<1,1,0, computeStream>>>(d_totalRootChild);

    // Find un_safe components
	Find_Unsafe_Component<<<no_of_blocks, totalThreads, 0, computeStream>>>(
		root, 
		no_of_vertices, 
		d_rep, 
		d_isSafe, 
		d_cutVertex, 
		d_parent, 
		d_totalRootChild);

    CUDA_CHECK(cudaGetLastError(), "Find_Unsafe_Component Kernel launch failed");
    CUDA_CHECK(cudaStreamSynchronize(computeStream), "Failed to synchronize Find_Unsafe_Component");

    CUDA_CHECK(
    	cudaMemcpyAsync(
    		h_totalRootChild, 
    		d_totalRootChild, 
    		sizeof(int), 
    		cudaMemcpyDeviceToHost, 
    		transD2HStream
    	), 
    	"d_totalRootChild cannot be copied into cpu"
    );
    CUDA_CHECK(cudaStreamSynchronize(transD2HStream), "Failed to synchronize transD2HStream after Find_Unsafe_Component");

    if(!isFirstBatch) {
		updateCutVertex<<<no_of_blocks, totalThreads, 0, computeStream>>>(
			no_of_vertices, 			// input
			d_parent, 					// input
			d_partOfFundamental, 		// input
			d_offset, 					// input
			d_cutVertex 				// output
			);
    	CUDA_CHECK(cudaGetLastError(), "updateCutVertex Kernel launch failed");
	} else {
		// handle the first batch differently as we don't have csr yet
		bool *is_parent = g_bcg_ds.d_isFakeCutVertex; // reusing to save space
		// update nonLeaf nodes
		updateLeafNode<<<no_of_blocks, totalThreads, 0, computeStream>>>(
			no_of_vertices, 
			d_parent, 
			is_parent
			);
		CUDA_CHECK(cudaGetLastError(), "updateLeafNode Kernel launch failed");
		
		updateCutVertex_firstBatch<<<no_of_blocks, totalThreads, 0, computeStream>>>(
			no_of_vertices, 		// input
			d_parent, 				// input
			d_partOfFundamental, 	// input
			is_parent, 				// input
			d_cutVertex 			// output
			);
    	CUDA_CHECK(cudaGetLastError(), "updateCutVertex_firstBatch Kernel launch failed");
	}

    if (*h_totalRootChild > 1) {
    	set_root_cut_status<<<1, 1, 0, computeStream>>>(d_cutVertex, root);
    }

	if(isLastBatch) {
		repair(g_bcg_ds, root, child_of_root, *h_totalRootChild, computeStream);
		std::cout <<"repairing last batch done.\n";
		return; // Exit the function after repairing the pbcc graph
	}

	implicit_bcc<<<no_of_blocks, totalThreads, 0, computeStream>>>(
		no_of_vertices, 
		d_isSafe, 
		d_rep, 
		d_parent, 
		d_baseVertex, 
		d_nonTreeEdgeId, 
		d_imp_bcc_num);

    CUDA_CHECK(cudaGetLastError(), "implicit_bcc Kernel launch failed");
    
    if (*h_totalRootChild == 1) {
        update_root_bcc_number<<<1, 1, 0, computeStream>>>(d_imp_bcc_num, root, child_of_root);
    }

    // Final synchronization of the stream to ensure all operations are complete
    CUDA_CHECK(cudaStreamSynchronize(computeStream), 
    	"Failed to synchronize computeStream after all operations");
}

// ====[ End of assign_cut_vertex_BCC Code ]====