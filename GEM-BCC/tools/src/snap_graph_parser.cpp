/******************************************************************************
 * Simple program to parse snap_generated_graphs.
 *
 * To compile using the command line:
 *   g++ -std=c++17 -O3 snap_graph_parser.cpp -o snap_graph_parser -Iinclude/graph/ecl_graphs
 *
 ******************************************************************************/

#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib> // For exit()
#include <filesystem>

#include "ECLgraph.h"

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
std::string output_path = "/home/graphwork/cs22s501/datasets/ecl/generated_graphs/";

void print(const std::vector<std::set<int>>& adjlist) {
    int setNumber = 0; // Initialize set counter
    int ctr = 0;
    for (const auto &set : adjlist) {
        std::cout << "Vertex " << setNumber << ": ";
        for (const auto &element : set) {
            std::cout << element << " ";
            ctr++;
        }
        std::cout << std::endl; // New line after each set

        setNumber++; // Increment set counter
    }
}

void create_csr(const std::vector<std::set<int>>& adjlist, std::vector<long>& vertices, std::vector<int>& edges) {
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

void print_CSR(const std::vector<long>& vertices, const std::vector<int>& edges) {
    int numVertices = vertices.size() - 1;
    long ctr = 0;
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Vertex " << i << " is connected to: ";
        for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
            std::cout << edges[j] << " ";
            ctr++;
        }
        std::cout << "\n";
    }
    std::cout <<"edge count: " << ctr << std::endl;
}

std::string get_file_extension(std::string filename) {

    std::filesystem::path file_path(filename);

    // Extracting filename with extension
    filename = file_path.filename().string();
    // std::cout << "Filename with extension: " << filename << std::endl;

    // Extracting filename without extension
    std::string filename_without_extension = file_path.stem().string();
    // std::cout << "Filename without extension: " << filename_without_extension << std::endl;

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
    std::string filename;
    bool isOneIndexed = false;

    if (argc < 2) {
        std::cerr << "Error: No filename provided." << std::endl;
        std::cout << "Please enter the path to the input file: ";
        if (!(std::cin >> filename)) {
            std::cerr << "Error: Invalid filename." << std::endl;
            return 1; // Exit with error code
        }
        std::cout << "Is the data one-indexed? (0 for no, 1 for yes): ";
        if (!(std::cin >> isOneIndexed)) {
            std::cerr << "Error: Invalid input for indexing type." << std::endl;
            return 1; // Exit with error code
        }
    } else {
        filename = argv[1];
        if (argc >= 3) {
            std::string indexFlag = argv[2];
            if (indexFlag == "true" || indexFlag == "1") {
                isOneIndexed = true;
            } else if (indexFlag == "false" || indexFlag == "0") {
                isOneIndexed = false;
            } else {
                std::cerr << "Error: Invalid argument for indexing type. Use 'true/1' or 'false/0'." << std::endl;
                return 1; // Exit with error code
            }
        }
    }
    std::ifstream inputFile(filename);

    if (!inputFile) {
        std::cerr << "\nUnable to open the file: " << filename << std::endl;
        exit(EXIT_FAILURE); // Typically, returning non-zero from `main` indicates an error.
    }
    std::string graphType, line;
    int nodes;
    long numEdges;
    getline(inputFile, line);
    if (line[0] == '#') {
        // Extract filename
        std::size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            filename = line.substr(colonPos + 2); // skip ": "
        }

        // Check if the line contains graph type
        if (line.find("graph") != std::string::npos) {
            graphType = line.substr(2, line.find("graph") - 3);
        }
    }

    // Read and process header comments
    while (getline(inputFile, line)) {
        if (line[0] == '#') {

            // Check if the line contains node and edge information
            std::size_t pos = line.find("Nodes:");
            if (pos != std::string::npos) {
                std::size_t end = line.find("Edges:") - 1;
                nodes = std::stoi(line.substr(pos + 7, end - pos - 7));
                numEdges = std::stoi(line.substr(line.find("Edges:") + 7));
            }
        } else {
            break; // Exit the comment section
        }
    }

    // Output graph type and filename
    std::cout << "Filename: " << filename << std::endl;
    if (graphType.find("Directed") != std::string::npos) {
        std::cout << "Graph Type: Directed" << std::endl;
    } else if (graphType.find("Undirected") != std::string::npos) {
        std::cout << "Graph Type: Undirected" << std::endl;
    } else {
        std::cout << "Graph Type: Unknown" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Nodes: " << nodes << std::endl;
    std::cout << "Edges: " << numEdges << std::endl;
    std::vector<std::set<int>> adjlist(nodes);
    int u, v;
    std::istringstream iss(line);
    iss >> u >> v;
    if(isOneIndexed) {
        u--;
        v--;
    }
    adjlist[u].insert(v);
    adjlist[v].insert(u);
    
    while(getline(inputFile, line)) {
        iss.clear();
        iss.str(line);
        if(iss >> u >> v) { // Read edge data
            if(isOneIndexed) {
                u--;
                v--;
            }
            adjlist[u].insert(v);
            adjlist[v].insert(u); // Remove this line for directed graphs
        }
    }

    // print(adjlist);
    std::vector<long> vertices;
    std::vector<int> edges;
    create_csr(adjlist, vertices, edges);
    print_CSR(vertices, edges);
    output_path = output_path + get_file_extension(filename);
    writeECLgraph(vertices, edges, output_path);

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
