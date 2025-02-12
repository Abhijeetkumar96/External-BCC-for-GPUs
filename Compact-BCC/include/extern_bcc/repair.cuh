#ifndef REPAIR_H
#define REPAIR_H

#include "extern_bcc/bcg_memory_utils.cuh"

void repair(GPU_BCG& g_bcg_ds, const int& root, const int child_of_root, const int totalRootChild, cudaStream_t myStream);

#endif // REPAIR_H