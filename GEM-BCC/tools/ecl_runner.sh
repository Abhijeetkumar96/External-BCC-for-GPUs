#!/bin/bash

# This line indicates that the script should be run using the Bash shell.

# Array to hold process IDs
pids=()
# Array to hold filenames for each PID
pid_files=()

# The script starts a loop here. It will iterate over all files in a specified directory.
for file in /home/graphwork/cs22s501/datasets/txt/generated_graphs/modified/*; do
    
    # This if-statement checks if the current item in the loop is a regular file.
    if [ -f "$file" ]; then
        # Echo (print) a message stating which file is currently being processed.
        echo -e "\nProcessing $file..."

        # Run your program in the background. This allows the script to continue
        # running and not wait for this program to finish before continuing.
        # "./edge_list_to_ecl" is your executable, and "$file" is its argument.
        bin/edge_list_to_ecl "$file" &

        # "$!" is the process ID (PID) of the last background process started.
        # This PID is added to the 'pids' array.
        pids+=($!)
        # The corresponding filename is also stored in the 'pid_files' array.
        pid_files+=("$file")

        # Echo a message confirming that processing has started for the file.
        echo "Processing started for $file."
    fi
done

# After starting processes for all files, the script will now wait for each
# to finish and check their exit statuses.
for i in ${!pids[@]}; do
    # Extract the PID and corresponding filename for each process.
    pid=${pids[$i]}
    file=${pid_files[$i]}

    # 'wait $pid' waits for the process with the given PID to finish. 
    # The exit status of the process is then stored in 'status'.
    wait $pid
    status=$?

    # This if-else block checks the exit status.
    # - If 'status' is 0, it means the process finished successfully.
    # - If 'status' is not 0, it indicates an error occurred.
    if [ $status -eq 0 ]; then
        echo "Processing of $file completed successfully."
    else
        echo "Error occurred in processing $file. Exit status: $status."
    fi
done

# Echo a message to indicate that all files have been processed.
echo "All processing complete."
