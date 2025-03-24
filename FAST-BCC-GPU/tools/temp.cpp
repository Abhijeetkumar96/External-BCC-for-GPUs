// checking maximum possible sizes for GSH-webgraph.
#include <iostream>

int main() {
    int numVert = 988490691;
    long expected_edge = 33877399152;
    expected_edge *= 2;
    uint64_t* arr = nullptr;

    try {
        std::cout << "numVert: " << numVert << " and numEdges: " << expected_edge << std::endl;
        std::cout << "Expected Edges Size: " << expected_edge * sizeof(uint64_t) << std::endl;

        // csr data-structures
        std::cout << "Vertices array: " << numVert * sizeof(long) << std::endl;
        std::cout << "Edges array: " << expected_edge * sizeof(int) << std::endl;

        size_t size = numVert * sizeof(long) + expected_edge * sizeof(int);

        std::cout << "Total CSR expected memory: " << size << " Bytes.\n";

        // Try to allocate memory using new
        arr = new uint64_t[expected_edge];
        std::cout << "Memory allocation successful!" << std::endl;
    } catch (const std::bad_alloc& e) {
        // Catch memory allocation failure
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return 1; // Return an error code
    }

    delete[] arr;

    return 0;
}
