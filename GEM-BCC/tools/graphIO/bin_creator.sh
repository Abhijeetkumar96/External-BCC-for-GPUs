#!/bin/bash

# Paths to the input directories
DIRECTORY1="/raid/graphwork/datasets/large_graphs/txt/"
DIRECTORY2="/raid/graphwork/datasets/new_graphs/txt/"

OUTPUT_DIR1="/raid/graphwork/datasets/large_graphs/csr_bin"
OUTPUT_DIR2="/raid/graphwork/datasets/new_graphs/csr_bin/"

# Array to hold process IDs
pids=()

# Function to process files in a directory
process_directory() {
    local directory=$1
    local output_path=$2
    for file in "$directory"*.txt; do
        echo "Processing $file..."
        bin/edge_list_to_bin "$file" "$output_path" >> bin_out.log  &

        # Store the process ID of the background process
        pids+=($!)

        echo "Processing started for $file."
    done
}

# Process files in both directories
process_directory $DIRECTORY2 $OUTPUT_DIR2
process_directory $DIRECTORY1 $OUTPUT_DIR1

# Wait for all background processes to finish
for pid in ${pids[@]}; do
    wait $pid
done

echo "All processing complete."
