/*
Project Name: ECLgraph.h

This project is developed by Abhijeet for educational purposes. 
It is inspired by and based on the work originally created by Martin Burtscher at Texas State University.

Derived from "ECLgraph" by Martin Burtscher
Reference: https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/ECLgraph.h
*/

#ifndef ECL_GRAPH_H
#define ECL_GRAPH_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept> // For std::runtime_error

/**
 * Reads graph data from a binary file.
 *
 * @param filename The name of the file to read from.
 * @param vertices A reference to a vector where vertex information will be stored.
 * @param edges A reference to a vector where edge information will be stored.
 * @throws std::runtime_error if the file cannot be opened or read from.
 */
void readECLgraph(const std::string& filename, std::vector<long>& vertices, std::vector<int>& edges) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    // Reading sizes
    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    vertices.resize(size);
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    edges.resize(size);

    // Reading data
    inFile.read(reinterpret_cast<char*>(vertices.data()), vertices.size() * sizeof(long));
    inFile.read(reinterpret_cast<char*>(edges.data()), edges.size() * sizeof(int));
}

/**
 * Writes graph data to a binary file.
 *
 * @param vertices A const reference to a vector containing vertex information.
 * @param edges A const reference to a vector containing edge information.
 * @param filename The name of the file to write to.
 * @throws std::runtime_error if the file cannot be opened or written to.
 */
void writeECLgraph(const std::vector<long>& vertices, const std::vector<int>& edges, std::string filename) {
    
    std::cout << "Binary filename: " << filename << std::endl;
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Error opening file for writing: " + filename);
    }

    // Writing sizes first for easy reading
    size_t size = vertices.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    size = edges.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Writing data
    outFile.write(reinterpret_cast<const char*>(vertices.data()), vertices.size() * sizeof(long));
    outFile.write(reinterpret_cast<const char*>(edges.data()), edges.size() * sizeof(int));

    // Check for write errors
    if (!outFile.good()) {
        throw std::runtime_error("Error writing to file: " + filename);
    }
}

#endif // ECL_GRAPH_H
