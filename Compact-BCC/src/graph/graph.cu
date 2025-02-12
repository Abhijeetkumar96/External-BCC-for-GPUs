#include <fstream>
#include <cassert>
#include <omp.h>
#include <cuda_runtime.h>

#include "graph/graph.cuh"
#include "utility/utility.hpp"
#include "utility/cuda_utility.cuh"

undirected_graph::undirected_graph(const std::string& filename) : filepath(filename) {
    try {
        auto start = std::chrono::high_resolution_clock::now();
        readGraphFile();

        // allocate parent & level array (pinned memory)
        size_t bytes = numVert * sizeof(int);

        CUDA_CHECK(cudaMallocHost((void**)&p_parent,  bytes),  "Failed to allocate pinned memory for parent");
        CUDA_CHECK(cudaMallocHost((void**)&p_level,   bytes),  "Failed to allocate pinned memory for level");

        auto end = std::chrono::high_resolution_clock::now();
        read_duration = end - start;
    }   
    catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            throw;
    }
}

void undirected_graph::readGraphFile() {
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("File does not exist: " + filepath.string());
    }

    std::string ext = filepath.extension().string();
    if (ext == ".edges" || ext == ".eg2" || ext == ".txt") {
        readEdgeList();
    }
    else if (ext == ".mtx") {
        readMTXgraph();
    }
    else if (ext == ".gr") {
        readMETISgraph();
    }
    else if (ext == ".egr" || ext == ".bin" || ".csr") {
        readECLgraph();
    }
    else {
        throw std::runtime_error("Unsupported graph format: " + ext);
    }
}

void undirected_graph::readEdgeList() {
    // std::cout << "Reading edges file: " << getFilename() << std::endl;
    std::ifstream inFile(filepath);
    if (!inFile) {
        throw std::runtime_error("Error opening file: ");
    }
    inFile >> numVert >> numEdges;

    // Allocate host pinned memories
    size_t bytes = (numEdges/2) * sizeof(int);

    // Host pinned memory ds
    CUDA_CHECK(cudaMallocHost((void**)&src,  bytes),  "Failed to allocate pinned memory for src");
    CUDA_CHECK(cudaMallocHost((void**)&dest, bytes),  "Failed to allocate pinned memory for dest");

    long ctr = 0;
    std::vector<std::vector<int>> adjlist(numVert);
    int u, v;
    for(long i = 0; i < numEdges; ++i) {
        inFile >> u >> v;
        adjlist[u].push_back(v);
        if(u < v) {
            src[ctr] = u;
            dest[ctr] = v;
            ctr++;
        }
    }
    assert(ctr == numEdges/2);
    create_csr(adjlist);
}

void undirected_graph::readMTXgraph() {
    // std::cout << "Reading mtx file: " << getFilename() << std::endl;
    std::ifstream inFile(filepath);
    if (!inFile) {
        throw std::runtime_error("Error opening file: ");
    }
}

void undirected_graph::readMETISgraph() {
    // std::cout << "Reading metis file: " << getFilename() << std::endl;
    std::ifstream inFile(filepath);
    if (!inFile) {
        throw std::runtime_error("Error opening file: ");
    }
}   

void undirected_graph::readECLgraph() {
    // std::cout << "Reading ECL file: " << getFilename() << std::endl;

    std::ifstream inFile(filepath, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Error opening file: ");
    }

    // Reading sizes
    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    vertices.resize(size);
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    edges.resize(size);

    // Reading data
    inFile.read(reinterpret_cast<char*>(vertices.data()), vertices.size() * sizeof(long));
    inFile.read(reinterpret_cast<char*>(edges.data()), edges.size() * sizeof(int));

    numVert = vertices.size() - 1;
    numEdges = edges.size();

    csr_to_coo();
}

void undirected_graph::print_CSR() {
    for (int i = 0; i < numVert; ++i) {
        std::cout << "Vertex " << i << " is connected to: ";
        for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
            std::cout << edges[j] << " ";
        }
        std::cout << "\n";
    }
}

void undirected_graph::print_edgelist() {
    for(int i = 0; i < numEdges/2; ++i) {
        std::cout << "(" << src[i] << ", " << dest[i] << ") \n";
    }
    std::cout << std::endl;
}

void undirected_graph::basic_stats(const long& maxThreadsPerBlock, const bool& g_verbose, const bool& checker) {
    
    const std::string border = "========================================";

    std::cout << border << "\n"
    << "       Graph Properties & Execution Settings Overview\n"
    << border << "\n\n"
    << "Graph reading and CSR creation completed in " << formatDuration(read_duration.count()) << "\n"
    << "|V|: " << getNumVertices() << "\n"
    << "|E|: " << getNumEdges() / 2 << "\n"
    << "Average Degree: " << getAvgOutDegree() << "\n"
    << "Max Degree Vertex: " << getMaxDegreeVert() <<"\n"
    << "maxThreadsPerBlock: " << maxThreadsPerBlock << "\n"
    << "Verbose Mode: "     << (g_verbose ? "✅" : "❌") << "\n"
    << "Checker Enabled: "  << (checker ?   "✅" : "❌") << "\n"
    << border << "\n\n";
}

void undirected_graph::create_csr(const std::vector<std::vector<int>>& adjlist) {
    vertices.push_back(edges.size());
    for (int i = 0; i < numVert; i++) {
        edges.insert(edges.end(), adjlist[i].begin(), adjlist[i].end());
        vertices.push_back(edges.size());
    }

    long max_degree = 0;
    max_degree_vert = -1;
    avg_out_degree = 0.0;
    for (int i = 0; i < numVert; ++i) {
        long degree = vertices[i+1] - vertices[i];
        avg_out_degree += (double)degree;
        if (degree > max_degree) {
            max_degree = degree;
            max_degree_vert = i;
        }
    }
    avg_out_degree /= (double)numVert;
    
    assert(max_degree_vert >= 0);
    assert(avg_out_degree >= 0.0);
}

void undirected_graph::csr_to_coo() {

    // Allocate host pinned memories
    size_t bytes = (numEdges/2) * sizeof(int);

    // Host pinned memory ds
    CUDA_CHECK(cudaMallocHost((void**)&src,  bytes),  "Failed to allocate pinned memory for src");
    CUDA_CHECK(cudaMallocHost((void**)&dest, bytes),  "Failed to allocate pinned memory for dest");

    long ctr = 0;

    for (int i = 0; i < numVert; ++i) {
        for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
            if(i < edges[j]) {
                src[ctr]  = i;
                dest[ctr] = edges[j];
                ctr++;
            }
        }
    }    

    assert(ctr == numEdges/2);

    long max_degree = 0;
    max_degree_vert = -1;
    avg_out_degree = 0.0;
    for (int i = 0; i < numVert; ++i) {
        long degree = vertices[i+1] - vertices[i];
        avg_out_degree += (double)degree;
        if (degree > max_degree) {
            max_degree = degree;
            max_degree_vert = i;
        }
    }
    avg_out_degree /= (double)numVert;
    
    assert(max_degree_vert >= 0);
    assert(avg_out_degree >= 0.0);
}

undirected_graph::~undirected_graph() {
    // free host pinned memories
    if(src) CUDA_CHECK(cudaFreeHost(src),                   "Failed to free pinned memory for src");
    if(dest) CUDA_CHECK(cudaFreeHost(dest),                 "Failed to free pinned memory for dest");

    if (p_parent) CUDA_CHECK(cudaFreeHost(p_parent),        "Failed to free pinned memory for parent");
    if (p_level) CUDA_CHECK(cudaFreeHost(p_level),          "Failed to free pinned memory for level");

    CUDA_CHECK(cudaDeviceReset(),                           "Failed to reset device");
}