#!/bin/bash

# Run txt_graph_parser
./txt_graph_parser /raid/graphwork/new/original/MOLIERE_2016.txt

# Check if the first command was successful
if [ $? -eq 0 ]; then
    echo "txt_graph_parser executed successfully."
else
    echo "txt_graph_parser failed."
    exit 1
fi

# Run edgelist_to_bin
./edgelist_to_bin /raid/graphwork/new/MOLIERE_2016.txt

# Check if the second command was successful
if [ $? -eq 0 ]; then
    echo "edgelist_to_bin executed successfully."
else
    echo "edgelist_to_bin failed."
    exit 1
fi
