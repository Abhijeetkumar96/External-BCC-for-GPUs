#include "multicore_bfs/multicore_bfs.hpp"
#include "multicore_bfs/thread.hpp"

#include <stdexcept>
#include <iostream>
#include <vector>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#define ALPHA 15.0
#define BETA 24.0

#define debug 0 

// Constructor implementation
BFS::BFS(const int* edges, const long* vertices, int* _parent, int* _level, 
         const int numVert, const long numEdges, const double _avg_out_degree, const int _root)
    : out_array(edges), 
      out_degree_list(vertices), 
      num_verts(numVert), 
      num_edges(numEdges), 
      avg_out_degree(_avg_out_degree), 
      root(_root), 
      h_parent(_parent), 
      h_level(_level) 
{
    init();
}

void BFS::init() {
    std::cout << "Doing MULTICORE-BFS init" << std::endl;
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < num_verts; ++i)
            h_level[i] = -1;
        #pragma omp for nowait
        for (int i = 0; i < num_verts; ++i)
            h_parent[i] = -1;
    }
}

void BFS::start() {

    int* parents = h_parent;
    int* levels = h_level;

    #ifdef _OPENMP
        std::cout << "OpenMP is enabled. Version: " << _OPENMP << std::endl;
    #else
        std::cout << "OpenMP is not enabled." << std::endl;
    #endif

    int* queue = new int[num_verts];
    int* queue_next = new int[num_verts];
    int queue_size = 0;  
    int queue_size_next = 0;

    queue[0] = root;
    queue_size = 1;
    parents[root] = root;
    levels[root] = 0;

    int level = 1;
    int num_descs = 0;
    int local_num_descs = 0;
    bool use_hybrid = false;
    bool already_switched = false;

    #pragma omp parallel
    {
        int thread_queue[ THREAD_QUEUE_SIZE ];
        int thread_queue_size = 0;

        while (queue_size)
        {

            if (!use_hybrid)
            {
                #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
                for (int i = 0; i < queue_size; ++i)
                {
                    int vert = queue[i];
                    long out_degree = out_degree_list[vert+1] - out_degree_list[vert];
                    const int* outs = &out_array[out_degree_list[vert]];
                    for (long j = 0; j < out_degree; ++j)
                    {      
                        int out = outs[j];
                        if (levels[out] < 0)
                        {
                            levels[out] = level;
                            parents[out] = vert;
                            ++local_num_descs;
                            add_to_queue(thread_queue, thread_queue_size, queue_next, queue_size_next, out);
                        }
                    }
                }
            }
            else
            {
                int prev_level = level - 1;

                #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
                for (int vert = 0; vert < num_verts; ++vert)
                {
                    if (levels[vert] < 0)
                    {
                        long out_degree = out_degree_list[vert+1] - out_degree_list[vert];
                        const int* outs = &out_array[out_degree_list[vert]];
                        for (long j = 0; j < out_degree; ++j)
                        {
                            int out = outs[j];
                            if (levels[out] == prev_level)
                            {
                                levels[vert] = level;
                                parents[vert] = out;
                                ++local_num_descs;
                                add_to_queue(thread_queue, thread_queue_size, queue_next, queue_size_next, vert);
                                break;
                            }
                        }
                    }
                }
            }
    
            empty_queue(thread_queue, thread_queue_size, queue_next, queue_size_next);
            #pragma omp barrier

            #pragma omp single
            {
                if (debug)
                  std::cout << "num_descs: " << num_descs << " local: " << local_num_descs << std::endl;
                num_descs += local_num_descs;

                if (!use_hybrid)
                {  
                    double edges_frontier = (double)local_num_descs * avg_out_degree;
                    double edges_remainder = (double)(num_verts - num_descs) * avg_out_degree;
                    if ((edges_remainder / ALPHA) < edges_frontier && edges_remainder > 0 && !already_switched)
                    {
                        if (debug)
                            std::cout << "\n=======switching to hybrid\n\n";

                        use_hybrid = true;
                    }
                    if (debug)
                        std::cout << "edge_front: " << edges_frontier << ", edge_rem: " << edges_remainder << std::endl;
                }
                else
                {
                    if ( ((double)num_verts / BETA) > local_num_descs  && !already_switched)
                    {
                        if (debug)
                            std::cout << "\n=======switching back\n\n";

                        use_hybrid = false;
                        already_switched = true;
                    }
                }
                local_num_descs = 0;

                queue_size = queue_size_next;
                queue_size_next = 0;
                int* temp = queue;
                queue = queue_next;
                queue_next = temp;
                ++level;
            } // end single
        }
    } // end parallel


    if (debug)
        std::cout <<"Final num desc: " << num_descs << std::endl;
  
    delete [] queue;
    delete [] queue_next;

}

// Implementation of verify_spanning_tree
bool BFS::verify() {

    int* comp_num = new int[num_verts];
    int num_comp = num_verts;
    int* parent_ptr = h_parent;

    std::vector<std::vector<int>> vertices_in_comp(num_verts);

    for (int i = 0; i < num_verts; ++i) {
        comp_num[i] = i;
        vertices_in_comp[i].push_back(i);
    }

    for (int i = 0; i < num_verts; ++i) {
        if (i != root) {
            if (comp_num[i] != comp_num[parent_ptr[i]]) {
                int u = comp_num[i];
                int v = comp_num[parent_ptr[i]];
                int small, big;
                if (vertices_in_comp[u].size() < vertices_in_comp[v].size()) {
                    small = u;
                    big = v;
                }
                else {
                    small = v;
                    big = u;
                }

                for (size_t k = 0; k < vertices_in_comp[small].size(); ++k) {
                    int ver = vertices_in_comp[small][k];
                    comp_num[ver] = big;
                    vertices_in_comp[big].push_back(ver);
                }

                vertices_in_comp[small].clear();
                --num_comp;
            }
        }
    }

    delete[] comp_num;

    if (num_comp == 1) 
        return true;
    else 
    return false;
}

void BFS::print(const int* arr) {
    int j = 0;
    for(int i = 0; i < num_verts; ++i) {
        std::cout << j++ <<" : " << arr[i] << std::endl;
    }
    std::cout << std::endl;
}
