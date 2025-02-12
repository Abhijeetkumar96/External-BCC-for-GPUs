#ifndef EXTERN_BCC_H
#define EXTERN_BCC_H

#include "extern_bcc/bcg_memory_utils.cuh"

// void extern_bcc(undirected_graph& g, BFS& bfs, GPU_BCG& g_bcg_ds);
void extern_bcc(GPU_BCG& g_bcg_ds);

#endif // EXTERN_BCC_H