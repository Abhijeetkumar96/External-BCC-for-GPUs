#ifndef EXTERN_BCC_H
#define EXTERN_BCC_H

#include "graph/graph.cuh"
#include "multicore_bfs/multicore_bfs.hpp"
#include "extern_bcc/bcg_memory_utils.cuh"


void extern_bcc(undirected_graph& g, BFS& bfs, GPU_BCG& g_bcg_ds);

#endif // EXTERN_BCC_H