#ifndef MULTICORE_BFS_H
#define MULTICORE_BFS_H

#include <vector>

class BFS {
private:

    const int* out_array;           // Pointer to edges array
    const long* out_degree_list;    // Pointer to vertices array
    int num_verts;                  // Number of vertices
    long num_edges;                 // Number of edges (total_num_edges / 2)
    double avg_out_degree;
    int root;
    int* h_parent;
    int* h_level;

public:

    BFS(const int* edges, const long* vertices, int* _parent, int* _level, const int numVert, const long numEdges, const double avg_out_degree, const int root);
    
    void init();
    void start(); 
    bool verify();
    void print(const int* arr);

    // Getter for root
    int getRoot() const {
        return root;
    }

    // print functions
    void print_parent() {
        print(h_parent);
    }

    void print_level() {
        print(h_level);
    }
};

#endif // MULTICORE_BFS_H
