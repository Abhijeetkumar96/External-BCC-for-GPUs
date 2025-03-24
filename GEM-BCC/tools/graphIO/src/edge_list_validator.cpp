#include <set>
#include <omp.h>
#include <queue>
#include <vector>
#include <chrono>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

void read_input(std::vector<std::set<int>>& newAdj, int& numVert, long& numEdges, std::string filename) {
    std::ifstream inFile(filename);
    if(!inFile) {
        std::cerr <<"Unable to open file for reading.\n";
        return;
    }

    inFile >> numVert >> numEdges;
    newAdj.resize(numVert);
    int u, v;
    for(long i = 0; i < numEdges; ++i) {
        inFile >> u >> v;
        newAdj[u].insert(v);
        newAdj[v].insert(u);
    }
}

int main(int argc, char* argv[]) {
	std::ios_base::sync_with_stdio(false);
	if(argc < 2) {
		std::cerr <<"Usage : " << argv[0] <<"<filename> " << std::endl;
		return EXIT_FAILURE;
	}
	std::string filename = argv[1];
	std::vector<std::set<int>> adjlist;
	int numVert;
    long numEdges;

    read_input(adjlist, numVert, numEdges, filename);

	long degree = 0;
	for(size_t i = 0; i < adjlist.size(); ++i) {
	    // Add the number of edges connected to vertex i
	    degree += adjlist[i].size();
	}
    assert(degree == numEdges);

    if(degree == numEdges) {
        std::cout << "Assertion passed: degree is twice the number of edges." << std::endl;
    } else {
        std::cerr << "Assertion failed: degree is not twice the number of edges." << std::endl;
        std::abort(); // Optionally abort the program, similar to assert failure
    }

	return EXIT_SUCCESS;
}