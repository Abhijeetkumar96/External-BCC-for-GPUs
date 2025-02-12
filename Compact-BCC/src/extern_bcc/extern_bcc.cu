//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <omp.h>
#include <vector>
#include <cassert>
#include <numeric>
#include <iterator>
#include <iostream>
#include <functional>

//---------------------------------------------------------------------
// Graph data-structure
//---------------------------------------------------------------------
#include "graph/graph.cuh" 

//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "utility/timer.hpp"             
#include "utility/utility.hpp"           
#include "utility/cuda_utility.cuh"      

//---------------------------------------------------------------------
// BFS and Biconnected Components (BCC) related functions
//---------------------------------------------------------------------
#include "extern_bcc/bcg.cuh"
#include "extern_bcc/extern_bcc.cuh"

void extern_bcc(undirected_graph& g, BFS& bfs, GPU_BCG& g_bcg_ds) {

    Timer myTimer;
    bfs.start();
    auto dur = myTimer.stop();
    std::cout <<"\nbfs finished in " << dur <<" ms.\n\n";
    
    if(checker) {
        if(bfs.verify()) 
            std::cout << "\n\nSpanning tree successfully verified.\n";
        else 
            std::cout << "Warning: Parent array does not represent a valid spanning tree.\n";
    }

    if(g_verbose) {
        std::cout << "Parent Array starts:" << "\n";
        bfs.print_parent();
        std::cout << "Parent Array ends." << "\n";
        std::cout << "Level Array starts:" << "\n";
        bfs.print_level();
        std::cout << "Level Array ends." << "\n";
    }

    int numVert     =   g.getNumVertices();
    long numEdges   =   g.getNumEdges() / 2;

    const int* parent = g.getParent();
    const int* level  = g.getLevel();

    // edge-stream
    g_bcg_ds.src    = g.getSrc();
    g_bcg_ds.dest   = g.getDest();

    // csr data-structures
    const std::vector<long>& vertices    = g.getVertices(); // vertex offset
    const std::vector<int>&  edges       = g.getEdges(); //edge list

    // Copy parent and level array to device using stream
    CUDA_CHECK(
        cudaMemcpyAsync(
            g_bcg_ds.d_parent, 
            parent, 
            numVert * sizeof(int), 
            cudaMemcpyHostToDevice,
            g_bcg_ds.transH2DStream
            ), 
        "Failed to copy parent to device"
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            g_bcg_ds.d_level, 
            level, 
            numVert * sizeof(int), 
            cudaMemcpyHostToDevice,
            g_bcg_ds.transH2DStream
            ), 
        "Failed to copy level to device"
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            g_bcg_ds.d_org_parent, 
            g_bcg_ds.d_parent, 
            numVert * sizeof(int), 
            cudaMemcpyDeviceToDevice,
            g_bcg_ds.computeStream
            ), 
        "Failed to copy parent from device to device"
    );

    g_bcg_ds.root = bfs.getRoot();
    g_bcg_ds.h_child_of_root[0] = edges[vertices[g_bcg_ds.root]];

    CUDA_CHECK(cudaStreamSynchronize(g_bcg_ds.transH2DStream), "Failed to synchronize transH2DStream after all operations");
    CUDA_CHECK(cudaStreamSynchronize(g_bcg_ds.computeStream), "Failed to synchronize computeStream after all operations");

    myTimer.reset();
    computeBCG(g_bcg_ds);

    dur = myTimer.stop();
    std::cout <<"computeBCG finished in: " << dur <<" ms." << std::endl;
}
