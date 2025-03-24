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
long find_connected_comp(const std::vector<std::vector<int>>& adjList) {
    int numNodes = adjList.size();
    std::vector<bool> visited(numNodes, false);
    long numComponents = 0;
    for (int node = 0; node < numNodes; ++node) {
        if (!visited[node]) {
            bfs(node, adjList, visited);
            numComponents++;
        }
    }

    return numComponents;
}

std::string get_filename_without_extension(const std::string& filename) {
    std::filesystem::path file_path(filename);
    return file_path.stem().string();
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

    int cc = find_connected_comp(adjList);

    std::cout << "Number of CC in " << get_filename_without_extension(filename) << " is: " << cc << std::endl;

    return 0;
}
