/**
 * @file EdgeAdderForConnectivity.cpp
 * @brief Finds the number of connected components in a graph and connects them if more than one.
 * 
 * This program takes a graph, represented as a set of edges and vertices, and identifies
 * all connected components. If the graph is not fully connected (i.e., has more than one
 * connected component), the program will add the minimum number of edges required to make
 * the graph fully connected.
 * 
 * @author Abhijeet
 * @date Nov 18, 2023
 */

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>

// #define DEBUG

std::string output_path;

std::string get_file_wo_ext(const std::string& filename) {
    std::filesystem::path file_path(filename);

    // Extracting filename with extension
    std::string filename_with_ext = file_path.filename().string();

    return filename_with_ext;
}

// Function to perform breadth-first search (BFS)
void bfs(long node, const std::vector<std::vector<long>>& adjList, std::vector<bool>& visited) {
    std::queue<long> q;
    q.push(node);
    visited[node] = true;

    while (!q.empty()) {
        long currNode = q.front();
        q.pop();

        for (long neighbor : adjList[currNode]) {
            if (!visited[neighbor]) {
                q.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}

// Function to find the number of connected components in the graph
long findConnectedComponents(const std::vector<std::vector<long>>& adjList, std::vector<std::pair<long, long>>& edgesToAdd) {
    long numNodes = adjList.size();
    std::vector<bool> visited(numNodes, false);
    long numComponents = 0;
    long prev = -1; // Start with an invalid value indicating no previous component yet
    for (long node = 0; node < numNodes; ++node) {
        if (!visited[node]) {
            if (prev != -1) { // Ensure this is not the first component
                edgesToAdd.push_back(std::make_pair(prev, node)); // Connect previous component to current
                edgesToAdd.push_back(std::make_pair(node, prev)); // Connect previous component to current
            }
            bfs(node, adjList, visited);
            numComponents++;
            prev = node; // Update prev to the current node for the next component
        }
    }

    return numComponents;
}

// Function to add new edges to the existing graph & update the numEdges
void makeFullyConnected(
    std::string filename, 
    const std::vector<std::pair<long, long>>& edgesToAdd,
    const std::vector<std::vector<long>>& adjList,
    long numVert, long numEdges) {

    numEdges += edgesToAdd.size();
    filename= output_path + get_file_wo_ext(filename);
    std::cout <<"output_path: " << filename << std::endl;
    std::ofstream outFile(filename);
    if(!outFile) {
        std::cerr <<"Unable to open file for writing.\n";
        return;
    }

    outFile << numVert <<" " << numEdges <<"\n";
    for(long i = 0; i < numVert; ++i) {
        for(long j = 0; j < adjList[i].size(); ++j) {
            outFile << i <<" " << adjList[i][j] <<"\n";
        }
    }

    for(long i = 0; i < edgesToAdd.size(); ++i) 
        outFile << edgesToAdd[i].first <<" " << edgesToAdd[i].second <<"\n";
}

void write_output(
    std::string filename, 
    const std::vector<std::vector<long>>& adjList,
    long numVert, long numEdges) {

    filename= output_path + get_file_wo_ext(filename);
    std::cout <<"output_path: " << filename << std::endl;
    std::ofstream outFile(filename);
    if(!outFile) {
        std::cerr <<"Unable to open file for writing.\n";
        return;
    }

    outFile << numVert <<" " << numEdges <<"\n";
    for(long i = 0; i < numVert; ++i) {
        for(long j = 0; j < adjList[i].size(); ++j) {
            outFile << i <<" " << adjList[i][j] <<"\n";
        }
    }
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    if(argc < 3) {
        std::cerr << "Usage : " << argv[0] <<" filename " <<" output_path" << std::endl;
        return 0;
    }

    std::string filename = argv[1];
    output_path = argv[2];

    std::ifstream inputFile(filename);

    if (inputFile.is_open()) {
        long numNodes, numEdges;
        inputFile >> numNodes >> numEdges;

        std::vector<std::vector<long>> adjList(numNodes);

        for (long i = 0; i < numEdges; ++i) {
            long u, v;
            inputFile >> u >> v;
            if(u == v)
                continue;
            adjList[u].push_back(v);
            // adjList[v].push_back(u);
        }

        // numEdges *= 2;

        inputFile.close();

        #ifdef DEBUG
            for(size_t i = 0; i < adjList.size(); ++i) {
                std::cout << i <<" : ";
                for(auto const& j : adjList[i])
                    std::cout << j <<" ";
                std::cout << std::endl;
            }
        #endif

        std::vector<std::pair<long, long>> edgesToAdd;
        long numComponents = findConnectedComponents(adjList, edgesToAdd);

        if(numComponents > 1)
            makeFullyConnected(filename, edgesToAdd, adjList, numNodes, numEdges);
        else
            write_output(filename, adjList, numNodes, numEdges);

        #ifdef DEBUG
            std::cout <<"edgesToAdd size = " << edgesToAdd.size() << std::endl;
            for(const auto &i : edgesToAdd)
                std::cout << i.first <<" " << i.second << std::endl;
        #endif

        std::cout <<"In file : " << get_file_wo_ext(filename) << ", ";
        std::cout << "Number of connected components: " << numComponents << std::endl;
    } else {
        std::cout << "Unable to open the input file." << std::endl;
    }

    return 0;
}
