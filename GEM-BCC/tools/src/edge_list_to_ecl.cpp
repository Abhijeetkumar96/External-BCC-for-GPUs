/******************************************************************************
 * Simple program to parse txt files.
 *
 * To compile using the command line:
 *   g++ -std=c++17 -O3 edge_list_to_ecl.cpp -o edge_list_to_ecl
 *
 ******************************************************************************/

#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib> 
#include <filesystem>

#include "ECLgraph.h"

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
// std::string output_path = "/home/graphwork/cs22s501/datasets/ecl/generated_graphs/";
std::string output_path = "/raid/graphwork/datasets/large_graphs/csr_bin/";

/* 
    The vector.insert() function in C++ is used to insert elements into a vector. Here's a brief overview of how it works:

    iterator insert(iterator position, const T& value);
    position: An iterator that points to the position in the vector where the new element(s) should be inserted.
    value: The value to be inserted. This can be a single value, a range of values, or a number of copies of a value.
*/

// Creating CSR
// The 'vertices' vector stores indices indicating the start of each vertex's adjacency list in the 'edges' vector.
// The 'edges' vector concatenates all adjacency lists, providing efficient and compact storage of the graph.
void create_csr(const std::vector<std::vector<int>>& adjlist, std::vector<long>& vertices, std::vector<int>& edges) {

    int n = adjlist.size();
    vertices.push_back(edges.size());
    for (int i = 0; i < n; i++) {
        edges.insert(edges.end(), adjlist[i].begin(), adjlist[i].end());
        vertices.push_back(edges.size());
    }
}

std::string get_file_extension(std::string filename) {

    std::filesystem::path file_path(filename);

    // Extracting filename with extension
    filename = file_path.filename().string();
    
    // Extracting filename without extension
    std::string filename_without_extension = file_path.stem().string();
    
    return filename_without_extension;
}

bool validate_csr(const std::vector<long>& vertices1, const std::vector<int>& edges1, 
                  const std::vector<long>& vertices2, const std::vector<int>& edges2) {
    return vertices1 == vertices2 && edges1 == edges2;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: No filename provided." << std::endl;
        return 1; // Exit with error code
    }

    std::string filename = argv[1];
    std::ifstream inputFile(filename);

    if (!inputFile) {
        std::cerr << "\nUnable to open the file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << output_path + get_file_extension(filename) << ".egr" << std::endl;
    int u, v;
    int numNodes;
    long numEdges;
    inputFile >> numNodes >> numEdges;
    std::vector<std::vector<int>> adjlist(numNodes);
    
    for(size_t i = 0; i < numEdges; ++i) {
        inputFile >> u >> v;
        adjlist[u].push_back(v);
    }

    std::vector<long> vertices;
    std::vector<int> edges;
    create_csr(adjlist, vertices, edges);

    output_path = output_path + get_file_extension(filename);
    writeECLgraph(vertices, edges, output_path + ".egr");

    // Reading the graph back for verification
    std::vector<long> read_vertices;
    std::vector<int> read_edges;
    readECLgraph(output_path + ".egr", read_vertices, read_edges);

    // Validate the CSR structure
    bool isValid = validate_csr(vertices, edges, read_vertices, read_edges);

    if (isValid) {
        std::cout << "Validation successful: CSR data matches." << std::endl;
        return 0; // Return success code
    } else {
        std::cerr << "Validation failed: CSR data does not match." << std::endl;
        return 1; // Return error code
    }

    return 0;
}
