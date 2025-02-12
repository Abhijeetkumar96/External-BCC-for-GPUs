#include <stxxl/vector>

int main() {
    const uint64_t num_elements = 33877399152; // Example size

    // STXXL Vector: stored on disk but behaves like an in-memory container
    stxxl::VECTOR_GENERATOR<uint64_t>::result vec(num_elements * 2);

    // Fill the vector with some data
    for (uint64_t i = 0; i < num_elements; ++i) {
        vec.push_back(i);
    }

    // Access some data
    std::cout << "Element at index 0: " << vec[0] << std::endl;

    return 0;
}
