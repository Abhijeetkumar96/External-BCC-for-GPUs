/******************************************************************************
* Functionality: Writing output to file
 ******************************************************************************/

#ifndef EXPORT_OUTPUT_CUH
#define EXPORT_OUTPUT_CUH

//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <omp.h>
#include <unistd.h>

#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <iterator>
#include <iostream>
#include <filesystem>
#include <functional>
#include <unordered_set>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// Utility functions
//---------------------------------------------------------------------
#include "extern_bcc/bcg.cuh"
#include "utility/cuda_utility.cuh"

//---------------------------------------------------------------------
// Debugger Flag
//---------------------------------------------------------------------
// #define DEBUG


class BCCOutputManager {
private:
    // 1. bcg related data - structures
    int bcg_numVert;
    long bcg_numEdges;
    
    // bcg edgeList
    std::vector<int> bcg_U;
    std::vector<int> bcg_V;

    // bcc info of the bcg graph
    std::vector<int> bcg_imp_bcc_num;
    std::vector<int> bcg_cut_vertex;
    std::vector<int> bcg_parent;
    std::vector<int> bcg_cut_ps;

    // mapping from original graph --> bcg graph
    std::vector<int> h_mapping;

    // 2. main graph related info
    const int numVert;
    const long numEdges;
    int ncv = 0;
    int nbcc = 0;
    
    // i. edge stream
    const int* U;
    const int* V;
    
    // output:
    // a. cut_vertex & bcc info
    std::vector<int> h_cut_vertex;
    std::vector<int> v_bcc_num;
    std::vector<int> e_bcc_num;

    // b. filenames
    const std::string& filename; 
    const std::string& output_path;

    // c. running time
    double dur;

    inline int get_edge_bcc(int, int);

public:
    BCCOutputManager(int nv, long ne, const int* u, const int* v, const GPU_BCG& g_bcg_ds, const std::string& _filename, const std::string& out_path, double& _dur);
    void update_vertex_cut_bcc();
    void assign_edge_labels();
    void write_to_file();
    void process_info();
    void write(const GPU_BCG& g_bcg_ds);
    void show_summary(const GPU_BCG& g_bcg_ds);
    void setup_dir(const std::string& path);
    void display_summary(int cut_vertices, int bccs);
};

// constructor
BCCOutputManager::BCCOutputManager(int nv, long ne, const int* u, const int* v, const GPU_BCG& g_bcg_ds, const std::string& _filename, const std::string& out_path, double& _dur)
    : numVert(nv), numEdges(ne), U(u), V(v), filename(_filename), output_path(out_path), dur(_dur)
{   

    bcg_numVert  = g_bcg_ds.numVert;
    bcg_numEdges = g_bcg_ds.numEdges;
    
    // 1. Update main graph related ds
    h_cut_vertex.resize(numVert);
    v_bcc_num.resize(numVert);
    e_bcc_num.resize(numEdges);

    bcg_U.resize(bcg_numEdges);
    bcg_V.resize(bcg_numEdges);

    bcg_parent.resize(bcg_numVert);
    h_mapping.resize(numVert);

    bcg_cut_vertex.resize(bcg_numVert);
    bcg_imp_bcc_num.resize(bcg_numVert);
    bcg_cut_ps.resize(bcg_numVert);

    // Ensure that the specified directory exists, creating it if absent.
    setup_dir(output_path);
}

void BCCOutputManager::write(const GPU_BCG& g_bcg_ds) {

    std::cout << "\n🖋️ Saving results to file, please stand by... ⌛" << std::endl;
    // -- bcg edge_list --
    CUDA_CHECK(cudaMemcpy(bcg_U.data(), g_bcg_ds.original_u,  bcg_numEdges * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back edge_u");
    CUDA_CHECK(cudaMemcpy(bcg_V.data(), g_bcg_ds.original_v,  bcg_numEdges * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back edge_v");

    // -- parent and mapping information -- 
    CUDA_CHECK(cudaMemcpy(bcg_parent.data(), g_bcg_ds.d_parent, bcg_numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back d_parent array");
    CUDA_CHECK(cudaMemcpy(h_mapping.data(),  g_bcg_ds.d_mapping, numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back h_mapping array");
    
    // -- cut_vertices and imp_bcc_num --
    CUDA_CHECK(cudaMemcpy(bcg_cut_vertex.data(), g_bcg_ds.d_cut_vertex, bcg_numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_cut_vertex to host");
    CUDA_CHECK(cudaMemcpy(bcg_imp_bcc_num.data(), g_bcg_ds.d_imp_bcc_num, bcg_numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_imp_bcc_num to host");
    
    std::exclusive_scan(bcg_cut_vertex.begin(), bcg_cut_vertex.end(), bcg_cut_ps.begin(), 0);

    update_vertex_cut_bcc();
    assign_edge_labels();
    write_to_file();
}

void BCCOutputManager::show_summary(const GPU_BCG& g_bcg_ds) {

    std::cout << "\n📊 Generating summary, please stand by... ⏳" << std::endl;
    // -- bcg edge_list --
    CUDA_CHECK(cudaMemcpy(bcg_U.data(), g_bcg_ds.original_u,  bcg_numEdges * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back edge_u");
    CUDA_CHECK(cudaMemcpy(bcg_V.data(), g_bcg_ds.original_v,  bcg_numEdges * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back edge_v");

    // -- parent and mapping information -- 
    CUDA_CHECK(cudaMemcpy(bcg_parent.data(), g_bcg_ds.d_parent, bcg_numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back d_parent array");
    CUDA_CHECK(cudaMemcpy(h_mapping.data(),  g_bcg_ds.d_mapping, numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back h_mapping array");
    
    // -- cut_vertices and imp_bcc_num --
    CUDA_CHECK(cudaMemcpy(bcg_cut_vertex.data(), g_bcg_ds.d_cut_vertex, bcg_numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_cut_vertex to host");
    CUDA_CHECK(cudaMemcpy(bcg_imp_bcc_num.data(), g_bcg_ds.d_imp_bcc_num, bcg_numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_imp_bcc_num to host");
    std::cout << "Im here.\n" << std::endl;
    #ifdef DEBUG
        std::cout << "BCG Cut Vertices info: " << "\n";
        for(auto i : bcg_cut_vertex)
            std::cout << i << " ";
        std::cout << std::endl;

        std::cout << "IBCC info: " << "\n";
        for(auto i : bcg_imp_bcc_num)
            std::cout << i << " ";
        std::cout << std::endl;
    #endif
    
    std::exclusive_scan(bcg_cut_vertex.begin(), bcg_cut_vertex.end(), bcg_cut_ps.begin(), 0);

    update_vertex_cut_bcc();
    assign_edge_labels();
    process_info();
}

void BCCOutputManager::update_vertex_cut_bcc() {
    #ifdef DEBUG
        std::cout <<"bcg_cut_vertex: ";
        for(auto i: bcg_cut_vertex)
            std::cout << i <<" ";
        std::cout << std::endl;

        std::cout <<"h_mapping: ";
        for(auto i: h_mapping)
            std::cout << i <<" ";
        std::cout << std::endl;
    #endif
    // happening correctly in parallel
    #pragma omp parallel for
    for(int i = 0; i < numVert; ++i) {
        int label_i = h_mapping[i];
        h_cut_vertex[i] = bcg_cut_vertex[label_i];
        if(!bcg_cut_vertex[label_i])
            v_bcc_num[i] = bcg_imp_bcc_num[label_i];
        else
            v_bcc_num[i] = bcg_imp_bcc_num[label_i] + bcg_cut_ps[label_i];
    }
}

inline int BCCOutputManager::get_edge_bcc(int u, int v) {
    // case_1: both are non-cut_vertices
    if(!bcg_cut_vertex[u] and !bcg_cut_vertex[v]) {
        return bcg_imp_bcc_num[u];
    }
    // case_2: both are cut_vertices
    else if(bcg_cut_vertex[u] and bcg_cut_vertex[v]) {
        // case a: if the edge is a tree edge then assign the bcc_num of child  
        if(bcg_parent[u] == v) {
            return bcg_imp_bcc_num[u];

        } else if(bcg_parent[v] == u) {
            return bcg_imp_bcc_num[v];
        }

        // case b: both are cut_vertices and nonTree edge
        else {
            return bcg_imp_bcc_num[u];
        }
    }

    // case_3: one vertex is cut_vertex and the other non_cut_vertex
    else {
        return bcg_imp_bcc_num[!bcg_cut_vertex[u]? u : v];
    }
}

void BCCOutputManager::assign_edge_labels() {
    #pragma omp parallel for
    for(long i = 0; i < numEdges; ++i) {
        int u = U[i];
        int v = V[i];
        int mapped_u = h_mapping[u];
        int mapped_v = h_mapping[v];
        
        // case 1: if the edge maps to an edge
        // std::cout <<"for u = " << u << " v = " << v <<" mapped_u = " << mapped_u <<" & mapped_v = " << mapped_v << std::endl;
        if(mapped_u != mapped_v)
            e_bcc_num[i] = get_edge_bcc(mapped_u, mapped_v);

        // case 2: if it maps to a vertex
        else {
            // case a: the vertex is not a cut vertex (directly assign that number)
            // case b: the vertex is a cut vertex (assign a new number)
            e_bcc_num[i] = !bcg_cut_vertex[mapped_u] ? bcg_imp_bcc_num[mapped_u] : bcg_numVert + bcg_cut_ps[mapped_u];
        }
    }
}

void BCCOutputManager::write_to_file() {

    std::string new_filename = filename + "_result.txt";
    new_filename = output_path + new_filename;
    
    std::ofstream outfile(new_filename);
    if(!outfile) {
        std::cerr <<"Unable to create file.\n";
        return;
    }

    outfile << numVert << "\t" << numEdges << "\n";
    int ncv, j; //ncv -> num_cut_vertex
    j = ncv = 0;
    outfile << "cut vertex status\n";
    for(const auto&i : h_cut_vertex) {
        if(i)
            ncv++;
        outfile << j++ << "\t" << i << "\n";
    }
    // std::cout <<"Total CV count: " << ncv << std::endl;

    j = 0;
    outfile << "vertex BCC number\n";
    for(const auto&i : v_bcc_num)
        outfile << j++ << "\t" << i << "\n";
    // write edge_bcc numbers
    j = 0;
    outfile << "edge BCC numbers\n";
    outfile << numEdges << "\n";
    for(long i = 0; i < numEdges; ++i) {
        outfile << U[i] <<" - " << V[i] << " -> " << e_bcc_num[i] << "\n";
    }

    int nbcc = 0;
    std::unordered_set<int> seen_bcc;

    for(int i = 0; i < numEdges; ++i){
        if( seen_bcc.find(e_bcc_num[i]) == seen_bcc.end() ){
            nbcc++;
            seen_bcc.insert(e_bcc_num[i]);
        }
    }

    outfile << ncv << " " << nbcc << "\n";

    display_summary(ncv, nbcc);
}

void BCCOutputManager::process_info() {

    int ncv; //ncv -> num_cut_vertex
    ncv = 0;

    for(const auto&i : h_cut_vertex) {
        if(i)
            ncv++;
    }

    int nbcc = 0;
    std::unordered_set<int> seen_bcc;

    for(int i = 0; i < numEdges; ++i){
        if( seen_bcc.find(e_bcc_num[i]) == seen_bcc.end() ){
            nbcc++;
            seen_bcc.insert(e_bcc_num[i]);
        }
    }

    display_summary(ncv, nbcc);
}

void BCCOutputManager::setup_dir(const std::string& path) {
    std::filesystem::path dir_path(path);
    if (!std::filesystem::exists(dir_path)) {
        if (std::filesystem::create_directories(dir_path)) {
            std::cout << "Directory created: " << path << "\n";
        } else {
            std::cerr << "Failed to create directory: " << path << "\n";
        }
    } 
    // else {
        // std::cout << "Directory already exists: " << path << std::endl;
    // }
}

void BCCOutputManager::display_summary(int cut_vertices, int bccs) {
    const std::string border = "========================================";
    std::cout << "\n";
    std::cout << border << "\n"
              << "      Graph Analysis Summary\n"
              << border << "\n\n"
              << "After thorough exploration and analysis of " << filename << " graph, we have identified:\n\n"
              << "- Articulation Points: " << cut_vertices << ". \n"
              << "- Biconnected Components (BCCs): " << bccs << ". \n\n"
              << " in " << formatDuration(dur) << " \n\n"
              << "Thank you for utilizing our graph analysis tool. We hope it has provided valuable insights into your dataset.\n"
              << border << "\n\n";

    std::cout << std::endl;
}

#endif // EXPORT_OUTPUT_CUH