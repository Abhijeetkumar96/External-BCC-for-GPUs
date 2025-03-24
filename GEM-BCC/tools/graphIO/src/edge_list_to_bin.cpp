/******************************************************************************
 * Simple program to parse snap_generated_graphs.
 *
 * To compile using the command line:
 *   g++ -std=c++17 -O3 edge_list_to_ecl.cpp -o edge_list_to_bin
 *
 ******************************************************************************/

#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>
#include <filesystem>

// #define DEBUG

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
std::string output_path;

std::string get_file_extension(std::string filename) {

    std::filesystem::path file_path(filename);

    // Extracting filename with extension
    filename = file_path.filename().string();
    
    // Extracting filename without extension
    std::string filename_without_extension = file_path.stem().string();
    
    return filename_without_extension;
}

void create_csr(
    const std::vector<std::set<int>>& adjlist, 
    std::vector<long>& vertices, 
    std::vector<int>& edges) {
    /* 
        The vector.insert() function in C++ is used to insert elements into a vector. Here's a brief overview of how it works:

        iterator insert(iterator position, const T& value);
        position: An iterator that points to the position in the vector where the new element(s) should be inserted.
        value: The value to be inserted. This can be a single value, a range of values, or a number of copies of a value.
    */

    // Creating CSR
    // The 'vertices' vector stores indices indicating the start of each vertex's adjacency list in the 'edges' vector.
    // The 'edges' vector concatenates all adjacency lists, providing efficient and compact storage of the graph.

    int n = adjlist.size();
    vertices.push_back(edges.size());
    for (int i = 0; i < n; i++) {
        edges.insert(edges.end(), adjlist[i].begin(), adjlist[i].end());
        vertices.push_back(edges.size());
    }
}

void write_bin_to_file(const std::vector<long>& vertices, const std::vector<int>& edges, std::string filename) {
    filename += ".egr";
    std::cout <<"binary filename: " << filename << std::endl;
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Writing sizes first for easy reading
    size_t size = vertices.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    size = edges.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Writing data
    outFile.write(reinterpret_cast<const char*>(vertices.data()), vertices.size() * sizeof(long));
    outFile.write(reinterpret_cast<const char*>(edges.data()), edges.size() * sizeof(int));
}

void readECLgraph(const std::string& filename, std::vector<long>& vertices, std::vector<int>& edges) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Error opening file: " + filename);
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
}

void print_csr(const std::vector<long>& vertices, const std::vector<int>& edges) {
    std::cout << "CSR Representation:" << std::endl;
    
    for (int i = 0; i < vertices.size() - 1; ++i) {
        std::cout << "Vertex " << i << " is connected to: ";
        for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
            std::cout << edges[j] << " ";
        }
        std::cout << "\n";
    }
}

bool validate_csr(const std::vector<long>& vertices1, const std::vector<int>& edges1, 
                  const std::vector<long>& vertices2, const std::vector<int>& edges2) {
    
    return vertices1 == vertices2 && edges1 == edges2;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: No filename provided." << std::endl;
        std::cerr <<"Usage: " <<"<filename> " <<" <output_path>\n";
        return 1; // Exit with error code
    }

    std::string filename = argv[1];
    output_path = argv[2];

    std::ifstream inputFile(filename);

    if (!inputFile) {
        std::cerr << "\nUnable to open the file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int u, v;
    int numNodes;
    long numEdges;
    inputFile >> numNodes >> numEdges;
    std::vector<std::set<int>> adjlist(numNodes);
    
    for(size_t i = 0; i < numEdges; ++i) {
        inputFile >> u >> v;
        adjlist[u].insert(v);
        adjlist[v].insert(u);
    }

    std::vector<long> vertices;
    std::vector<int> edges;
    create_csr(adjlist, vertices, edges);

    output_path = output_path + get_file_extension(filename);
    std::cout <<"output_path: " << output_path << std::endl;
    write_bin_to_file(vertices, edges, output_path);

    // Reading the graph back for verification
    std::vector<long> read_vertices;
    std::vector<int> read_edges;
    readECLgraph(output_path + ".egr", read_vertices, read_edges);

    // Validate the CSR structure
    bool isValid = validate_csr(vertices, edges, read_vertices, read_edges);

    #ifdef DEBUG 
        print_csr(read_vertices, read_edges);
    #endif

    if (isValid) {
        std::cout << "Validation successful: CSR data matches." << std::endl;
        return 0; // Return success code
    } else {
        std::cerr << "Validation failed: CSR data does not match." << std::endl;
        return 1; // Return error code
    }

    return 0;
}
