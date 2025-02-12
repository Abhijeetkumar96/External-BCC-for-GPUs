#include <iostream>
#include <cmath>    // Include cmath for std::ceil
#include <cuda_runtime.h> // Include CUDA runtime API

// CUDA error checking macro
#define CUCHECK(call, msg) {                                  \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
}

// CUDA kernel to print edges in the batch
__global__ 
void print_kernel(int* d_edges, int edge_count) {
    printf("Edges in device batch: ");
    for (int i = 0; i < edge_count; ++i) {
        printf("%d ", d_edges[i]);
    }
    printf("\n");
}

int main() {
    int num_edges = 100;
    int batch_size = 25;

    // Correct calculation for the number of batches
    int num_batches = std::ceil(static_cast<double>(num_edges) / batch_size);
    std::cout << "Number of batches: " << num_batches << std::endl;

    // Host memory allocation for edges
    int* h_edges = new int[num_edges];
    for (int i = 0; i < num_edges; ++i) {
        h_edges[i] = i; // Initialize host edges with values 0 to 129
    }

    // Device memory allocation for edges
    int* d_edges[2];
    for (int i = 0; i < 2; ++i) {
        CUCHECK(cudaMalloc(&d_edges[i], batch_size * sizeof(int)), "cudaMalloc error");
    }

    // Create CUDA streams
    cudaStream_t h2d_stream, compute_stream;
    CUCHECK(cudaStreamCreate(&h2d_stream), "cudaStreamCreate error for h2d_stream");
    CUCHECK(cudaStreamCreate(&compute_stream), "cudaStreamCreate error for compute_stream");

    int buffer_index = 0;

    // Initial asynchronous copy for the first batch
    int start = 0;
    int end = batch_size;
    if (end > num_edges) {
        end = num_edges;
    }
    int edge_count_in_batch = end - start;
    CUCHECK(cudaMemcpyAsync(d_edges[buffer_index], h_edges + start, edge_count_in_batch * sizeof(int), cudaMemcpyHostToDevice, h2d_stream), "cudaMemcpyAsync error");

    for (int i = 0; i < num_batches; ++i) {
        start = i * batch_size;
        end = (i + 1) * batch_size;

        // Ensure end doesn't exceed num_edges
        if (end > num_edges) {
            end = num_edges;
        }
        edge_count_in_batch = end - start;

        // Check if this is the last batch
        bool is_last_batch = (i == num_batches - 1);
        if (is_last_batch) {
            std::cout << "Processing the last batch." << std::endl;
        }

        std::cout << "Batch " << i << ": " << start << " to " << end - 1 << std::endl;
        std::cout << "Edges in batch: " << edge_count_in_batch << std::endl;

        int prev_index = buffer_index;
        // Toggle buffer index
        buffer_index = 1 - buffer_index;

        if (!is_last_batch) {
            // Copy edges to the device asynchronously on the h2d_stream for the next batch
            int next_start = (i + 1) * batch_size;
            int next_end = (i + 2) * batch_size;
            if (next_end > num_edges) {
                next_end = num_edges;
            }
            int next_edge_count = next_end - next_start;
            CUCHECK(cudaMemcpyAsync(d_edges[buffer_index], h_edges + next_start, next_edge_count * sizeof(int), cudaMemcpyHostToDevice, h2d_stream), "cudaMemcpyAsync error");
        }

        // Launch kernel to print edges on the compute stream
        print_kernel<<<1, 1, 0, compute_stream>>>(d_edges[prev_index], edge_count_in_batch);

        // Synchronize the compute stream to ensure kernel execution is completed
        // CUCHECK(cudaStreamSynchronize(compute_stream), "compute_stream error");

        std::cout << std::endl;
    }

    // Synchronize the H2D stream to ensure all memory transfers are complete
    CUCHECK(cudaStreamSynchronize(h2d_stream), "h2d_stream error");

    // Free device memory
    for (int i = 0; i < 2; ++i) {
        CUCHECK(cudaFree(d_edges[i]), "cudaFree error");
    }

    // Destroy CUDA streams
    CUCHECK(cudaStreamDestroy(h2d_stream), "cudaStreamDestroy error for h2d_stream");
    CUCHECK(cudaStreamDestroy(compute_stream), "cudaStreamDestroy error for compute_stream");

    // Free host memory
    delete[] h_edges;

    return 0;
}
