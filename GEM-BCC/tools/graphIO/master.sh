#!/bin/bash

# Make sure the scripts are executable
chmod +x edge_adder.sh
chmod +x bin_creator.sh

# Execute edge_adder.sh
echo "Executing edge_adder.sh"
./edge_adder.sh

# Check if edge_adder.sh executed successfully
if [ $? -eq 0 ]; then
    echo "edge_adder.sh executed successfully, proceeding with bin_creator.sh"
    # Execute bin_creator.sh
    ./bin_creator.sh

    # Check if bin_creator.sh executed successfully
    if [ $? -eq 0 ]; then
        echo "bin_creator.sh executed successfully"
    else
        echo "bin_creator.sh encountered an error"
    fi
else
    echo "edge_adder.sh encountered an error, stopping script execution"
fi
