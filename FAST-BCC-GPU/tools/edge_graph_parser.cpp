/******************************************************************************
 * Program to parse edge lists with configurable indexing (0-based or 1-based).
 *
 * To compile using the command line:
 *   g++ -std=c++17 -Wall -O3 edge_graph_parser.cpp -o edge_graph_parser
 *
 ******************************************************************************/

#include <algorithm> // For std::swap
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <set>

std::string output_path;

std::string get_file_wo_ext(const std::string& filename) {
    std::filesystem::path file_path(filename);

    // Extracting filename with extension
    std::string filename_with_ext = file_path.filename().string();

    return filename_with_ext;
}

void print(std::vector<std::vector<int>>& adjList) {
    for(int i = 0; i < adjList.size(); ++i) {
        std::cout << "neighbours of " << i << " : ";
        for (int j = 0; j < adjList[i].size(); ++j) {
            std::cout << adjList[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to perform breadth-first search (BFS)
void bfs(long node, const std::vector<std::vector<int>>& adjList, std::vector<bool>& visited) {
    std::queue<int> q;
    q.push(node);
    visited[node] = true;

    while (!q.empty()) {
        int currNode = q.front();
        q.pop();

        for (auto neighbor : adjList[currNode]) {
            if (!visited[neighbor]) {
                q.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}

// Function to find the number of connected components in the graph
long find_connected_comp(
    const std::vector<std::vector<int>>& adjList,
    std::vector<std::pair<int, int>>& edgesToAdd) {

    int numNodes = adjList.size();
    std::vector<bool> visited(numNodes, false);
    long numComponents = 0;
    long prev = -1; // Start with an invalid value indicating no previous component yet

    for (int node = 0; node < numNodes; ++node) {
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

std::string get_filename_without_extension(const std::string& filename) {
    std::filesystem::path file_path(filename);
    return file_path.stem().string();
}

// Function to add new edges to the existing graph & update the numEdges
void makeFullyConnected(
    std::string filename, 
    const std::vector<std::pair<int, int>>& edgesToAdd,
    const std::vector<std::vector<int>>& adjList,
    int numVert, long numEdges) {

    numEdges += edgesToAdd.size();
    filename= output_path + get_file_wo_ext(filename);
    std::cout <<"output_path: " << filename << std::endl;
    std::ofstream outFile(filename);
    if(!outFile) {
        std::cerr <<"Unable to open file for writing.\n";
        return;
    }

    outFile << numVert <<" " << numEdges <<"\n";
    for(int i = 0; i < numVert; ++i) {
        for(long j = 0; j < adjList[i].size(); ++j) {
            outFile << i <<" " << adjList[i][j] <<"\n";
        }
    }

    for(long i = 0; i < edgesToAdd.size(); ++i) 
        outFile << edgesToAdd[i].first <<" " << edgesToAdd[i].second <<"\n";
}

void write_output(
    std::string filename, 
    const std::vector<std::vector<int>>& adjList,
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

int main(int argc, char const* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    std::string filename = argv[1];

    std::string output_path = "/raid/graphwork/new/";
    std::string output_file = output_path + get_filename_without_extension(filename) + ".txt";

    // std::cout << "Output File: " << output_file << std::endl;

    std::ifstream inFile(filename);

    if (!inFile) {
        std::cerr << "Unable to open file for reading.\n";
        return 1;
    }

    std::set<std::pair<int, int>> edgelist;

    int u, v;
    int numVert = -1;
    long numEdges = 0;
    bool isOneIndex = true;

    while (inFile >> u >> v) {

        if(u == 0 or v == 0) {
            isOneIndex = false;
        }

        if(u == v)
            continue;

        // Ensure u <= v to avoid duplicate edges in undirected graph
        if (u > v)
            std::swap(u, v);

        edgelist.emplace(u, v);

        // Update the maximum vertex index
        if (u > numVert)
            numVert = u;
        if (v > numVert)
            numVert = v;

        numEdges++;
    }

    // inFile.close();

    std::cout << "Successfully read the file." << std::endl;
    std::cout << "isOneIndex: " << isOneIndex << std::endl;

    if(!isOneIndex)
        numVert++;

    std::vector<std::vector<int>> adjList(numVert);

    for (const auto& edge : edgelist) {
        int x = edge.first - isOneIndex;
        int y = edge.second - isOneIndex;
        
        adjList[x].push_back(y);
        adjList[y].push_back(x);
    }

    #ifdef DEBUG
        print(adjList);
    #endif

    std::vector<std::pair<int, int>> edgesToAdd;

    int cc = find_connected_comp(adjList, edgesToAdd);

    std::cout << "Number of CC in " << get_filename_without_extension(filename) << " is: " << cc << std::endl;

    if(cc > 1)
            makeFullyConnected(filename, edgesToAdd, adjList, numVert, numEdges);
        else
            write_output(filename, adjList, numVert, numEdges);

    return 0;
}
