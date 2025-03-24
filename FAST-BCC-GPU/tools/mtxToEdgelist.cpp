#include <set>
#include <omp.h>
#include <queue>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

class timer {
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
        std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
        bool running = false;
    public:
        timer() {
            start_time = std::chrono::high_resolution_clock::now();
            running = true;
        }

        void stop(const std::string& str) {
            end_time = std::chrono::high_resolution_clock::now();
            running = false;
            elapsedMilliseconds(str);
        }

        void elapsedMilliseconds(const std::string& str) {
            std::chrono::time_point<std::chrono::high_resolution_clock> end;
            if(running) {
                end = std::chrono::high_resolution_clock::now();
            } else {
                end = end_time;
            }
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time).count();
            std::cout << str <<" took " << duration << " ms." << std::endl;
        }
};

std::string get_file_extension(const std::string& filename) {
	size_t result;
    std::string fileExtension;
    result = filename.rfind('.', filename.size() - 1);
    if(result != std::string::npos)
        fileExtension = filename.substr(result + 1);
    return fileExtension;
}

int read_mtx(const std::string& filename, std::vector<uint64_t>& v) {
	std::ifstream inputFile(filename);
	if(!inputFile) {
		std::cerr <<"Unable to open file for reading." << std::endl;
		return -1;
	}
	std::string line;
	std::getline(inputFile, line);
	std::istringstream iss(line);
	std::string word;
	std::vector<std::string> words;
	
	while(iss >> word)
		words.push_back(word);

	if(words.size() != 5) {
		std::cerr <<"ERROR: mtx header is missing" << std::endl;
		return -1;
	}
    else 
        std::cout <<"MTX header read correctly.\n";

    std::getline(inputFile, line);
    while(line.size() > 0 && line[0]=='%') 
    	std::getline(inputFile,line);

    // Clear any error state flags
    iss.clear();

    // Set the new string for the istringstream
    iss.str(line);

    long row, col, num_of_entries;
    iss >> row >> col >> num_of_entries;

    if(row != col) {
        std::cerr<<"* ERROR: This is not a square matrix."<< std::endl;
        return -1;
    }
    v.reserve(num_of_entries);
    long entry_counter = 0;
    long ridx, cidx, value;

    while(!inputFile.eof() && entry_counter < num_of_entries) {
        getline(inputFile, line);
        entry_counter++;

        if (!line.empty()) {
            iss.clear();
            iss.str(line);
            iss >> ridx >> cidx; // >> value;
            ridx--;
            cidx--;

            if (ridx < 0 || ridx >= row)  
            	std::cout << "sym-mtx error: " << ridx << " row " << row << std::endl;

            if (cidx < 0 || cidx >= col)  
            	std::cout << "sym-mtx error: " << cidx << " col " << col << std::endl;

            if (ridx != cidx) {
                v.push_back(static_cast<uint64_t>(ridx) << 32 | (cidx));
                v.push_back(static_cast<uint64_t>(cidx) << 32 | (ridx));
            }
        }
    }
    long edges_count = v.size();
    std::cout << "Entry_counter: " << entry_counter << " and edges Count: " << edges_count << std::endl;

    return row;
}

// Function to remove duplicate elements 
void removeDuplicates(std::vector<uint64_t>& v) { 
    std::vector<uint64_t>::iterator it; 
  
    // using unique() method to remove duplicates 
    it = std::unique(v.begin(), v.end()); 
  
    // resize the new vector 
    v.resize(std::distance(v.begin(), it)); 
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

std::string get_filename_wo_ext(std::string path) {
    std::filesystem::path file_path(path);

    // Extracting filename with extension
    std::string filename = file_path.filename().string();
    std::cout << "Filename with extension: " << filename << std::endl;

    // Extracting filename without extension
    std::string filename_without_extension = file_path.stem().string();
    std::cout << "Filename without extension: " << filename_without_extension << std::endl;


    return filename_without_extension;
}

// BFS to visit all nodes in the connected component
void bfs(int start_node, const std::vector<long>& nindex, const std::vector<int>& nlist, std::vector<bool>& visited) {

    std::queue<int> q;
    q.push(start_node);
    visited[start_node] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        // Explore all neighbors of the current node
        for (long i = nindex[node]; i < nindex[node + 1]; ++i) {
            int neighbor = nlist[i];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

// Function to find the number of connected components using BFS
int find_cc(std::vector<uint64_t>& v,
    const std::vector<long>& nindex, 
    const std::vector<int>& nlist, 
    int nodes) {

    std::cout << "Inside find_cc." << std::endl;
    std::cout << "Number of nodes: " << nodes << std::endl;

    std::vector<bool> visited(nodes, false);
    int component_count = 0;
    long prev = -1; // Start with an invalid value indicating no previous component yet
    // Iterate through all nodes, and if unvisited, start BFS
    for (int node = 0; node < nodes; ++node) {
        if (!visited[node]) {
            if (prev != -1) { // Ensure this is not the first component
                #ifdef DEBUG
                    std::cout << "Adding the following edges: \n";
                    std::cout << prev << ", " << nodes;
                    std::cout << std::endl;
                #endif
                v.push_back(static_cast<uint64_t>(prev) << 32 | (node));
                v.push_back(static_cast<uint64_t>(node) << 32 | (prev));
            }

            bfs(node, nindex, nlist, visited);
            component_count++;
            prev = node;
        }
    }

    std::cout << "Total number of CC: " << component_count << std::endl;

    return component_count;
}

void read_graph(const std::string& filename, std::vector<uint64_t>& v, int& nodes, long& edges) {
	std::string ext = get_file_extension(filename);

	if (ext == "mtx") {
		nodes = read_mtx(filename, v);
	}
	else {
		std::cerr <<"Unsupported file format." << std::endl;
        exit(1);
    }

    if(nodes == -1) {
        return;
    }

    std::cout << "Graph Reading Completed." << std::endl;

    edges = v.size();
}

void create_CSR(
    std::vector<uint64_t>& v, 
    int nodes,
    std::vector<long>& vertices, 
    std::vector<int>& edges) {

    std::sort(v.begin(), v.end());
    std::cout << "Sorting Completed." << std::endl;
    removeDuplicates(v);
    std::cout << "Removing duplicates Completed." << std::endl;
    
    long edges_count = v.size();

    std::cout << "Number of Vertices: " << nodes << " and numEdges: " << edges_count << std::endl;

    vertices.resize(nodes + 1);
    edges.resize(edges_count);

    vertices[0] = 0;
    for (long i = 0; i < v.size(); i++) {
        uint64_t edge = v[i];

        int src  = edge >> 32;  // Extract higher 32 bits
        int dest = edge & 0xFFFFFFFF; // Extract lower 32 bits

        vertices[src + 1]++;  // Counting the number of edges for each node
        edges[i] = dest;    // Storing the destination node
    }

    // Convert `vertices` from counts to actual indices (cumulative sum)
    for (int i = 1; i <= nodes; i++) {
        vertices[i] += vertices[i - 1];
    }

    #ifdef DEBUG
        // Print the CSR representation
        std::cout << "CSR Graph Representation:" << std::endl;
        std::cout << "vertices (Row Pointers): ";
        for (int i = 0; i < vertices.size(); i++) {
            std::cout << vertices[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "edges (Column Indices): ";
        for (long i = 0; i < edges.size(); i++) {
            std::cout << edges[i] << " ";
        }
        std::cout << std::endl;

        // Print the edge list from CSR format
        std::cout << "\nEdge list:" << std::endl;
        for (int src = 0; src < nodes; src++) {
            for (long i = vertices[src]; i < vertices[src + 1]; i++) {
                std::cout << src << " -> " << edges[i] << std::endl;
            }
        }
    #endif
}

void create_binfile(std::vector<uint64_t>& v, int numVert, std::string filename) {

    std::vector<long> vertices;
    std::vector<int> edges;

    create_CSR(v, numVert, vertices, edges);

    std::string output_path = "/raid/graphwork/new_datasets/";
    output_path += get_filename_wo_ext(filename);
    std::cout <<"output_path: " << output_path << std::endl;
    write_bin_to_file(vertices, edges, output_path);

    // Reading the graph back for verification
    std::vector<long> read_vertices;
    std::vector<int> read_edges;
    readECLgraph(output_path + ".egr", read_vertices, read_edges);

    // Validate the CSR structure
    bool isValid = validate_csr(vertices, edges, read_vertices, read_edges);

    std::vector<uint64_t> v1;
    int num_CC = find_cc(v1,read_vertices, read_edges, read_vertices.size() - 1);

    #ifdef DEBUG 
        print_csr(read_vertices, read_edges);
    #endif

    if (isValid && num_CC == 1) {
        std::cout << "Validation successful: CSR data matches." << std::endl;
        return; // Return success code
    } else {
        std::cerr << "Validation failed: CSR data does not match." << std::endl;
        return; // Return error code
    }
}

void find_total_cc(std::vector<uint64_t>& v, int numVert, long numEdges) {
    std::vector<long> vertices;
    std::vector<int> edges;

    create_CSR(v, numVert, vertices, edges);
    int num_comp = find_cc(v, vertices, edges, numVert);
    std::cout << "Number of CC: " << num_comp << std::endl;
}

int main(int argc, char* argv[]) {
	std::ios_base::sync_with_stdio(false);
	if(argc < 2) {
		std::cerr <<"Usage : " << argv[0] <<"<filename> " << std::endl;
		return EXIT_FAILURE;
	}
	std::string filename = argv[1];
	timer read;
    std::vector<uint64_t> v;
    int numVert;
    long numEdges;
	read_graph(filename, v, numVert, numEdges);
	read.stop("Reading the graph");

    find_total_cc(v, numVert, numEdges);

    create_binfile(v, numVert, filename);

	return EXIT_SUCCESS;
}
