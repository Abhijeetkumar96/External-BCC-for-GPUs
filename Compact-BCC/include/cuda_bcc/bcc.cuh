#ifndef BCC_H
#define BCC_H

#include <vector>

#include "extern_bcc/bcg_memory_utils.cuh"

void cuda_bcc(GPU_BCG& g_bcg_ds, bool isLastBatch = false);
int update_bcc_numbers(GPU_BCG& g_bcg_ds, int numVert);
#endif // BCC_H