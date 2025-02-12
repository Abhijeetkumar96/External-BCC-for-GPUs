/**
 * CUDA Connected Components Finder
 * 
 * This header file contains the declaration of the connected components finding algorithm using CUDA, 
 * inspired by the work described in "Fast GPU Algorithms for Graph Connectivity" by Jyothish Soman, 
 * K. Kothapalli, and P. J. Narayanan, presented at the Large Scale Parallel Processing Workshop in IPDPS, 2010.
 * 
 * For more details, please refer original paper: https://github.com/jyosoman/GpuConnectedComponents
 *
 * The implementation computes the connected components in a graph using CUDA.
 * 
 * Parameters:
 *    int *d_uArr: Device pointer to an array of 'u' vertices of edges. 
 *    int *d_vArr: Device pointer to an array of 'v' vertices of edges.
 * 
 *    long numEdges: The number of edges in the graph.
 *    int numVert: The number of vertices in the graph.
 *    int *d_rep:  The output array.
 */

#include <set>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

void connected_comp(
	long numEdges, 
	int* u_arr, 
	int* v_arr, 
	int numVert, 
	int* d_rep, 
	int* h_flag,  		// pinned memory
	int* d_flag, 
	cudaStream_t computeStream,
	cudaStream_t transD2HStream);

#endif //CONNECTED_COMPONENTS_H
