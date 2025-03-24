#ifndef CUT_VERTEX_H
#define CUT_VERTEX_H

#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include "extern_bcc/bcg_memory_utils.cuh"

void assign_cut_vertex_BCC(GPU_BCG& g_bcg_ds, int root, int child_of_root, bool isFirstBatch = false, bool isLastBatch = false);

#endif // CUT_VERTEX_H