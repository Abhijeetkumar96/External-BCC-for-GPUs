#!/bin/bash

# Start resource monitoring in the background (logs memory, CPU usage every 10 seconds)
vmstat 10 >> system_monitor.log &

# Capture the PID of the vmstat process to stop it later
VMSTAT_PID=$!

OUTPUT_DIR="/raid/graphwork/datasets/web_graph/csr_bin/"

mkdir -p $OUTPUT_DIR

# Loop through each .edge file in the directory, skipping the excluded ones
for file in /raid/graphwork/new_datasets/edge_graph/*.edge; do
    # Skip the excluded files
    if [[ "$file" == *"clueweb12.edge" || "$file" == *"eu-2015.edge" || "$file" == *"gsh-2015.edge" ]]; then
        echo "Skipping $file" >> out.log
        continue
    fi
    
    echo "Processing $file" >> out.log
    
    # Run the edge graph parser and log output
    ./edge_graph_parser "$file" >> out.log
    ./edge_list_to_bin "$file" "$OUTPUT_DIR" >> bin_out.log
    
    # Capture the exit status of the parser
    EXIT_STATUS=$?
    
    # Check if the process was killed (137 is typical for OOM)
    if [ $EXIT_STATUS -eq 137 ]; then
        echo "Process for $file was killed due to OOM or external signal." >> out.log
    elif [ $EXIT_STATUS -eq 0 ]; then
        echo "Successfully completed for $file." >> out.log
    else
        echo "Process for $file exited with status $EXIT_STATUS." >> out.log
    fi
    
    echo "Completed processing $file" >> out.log
    echo "---------------------------------------------" >> out.log
done

# Stop the resource monitoring once all files are processed
kill $VMSTAT_PID
echo "Resource monitoring stopped." >> out.log
