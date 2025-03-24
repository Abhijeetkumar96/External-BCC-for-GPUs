#ifndef BCG_H
#define BCG_H

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#include "extern_bcc/bcg_memory_utils.cuh"

void computeBCG(GPU_BCG& g_bcg_ds);

#endif // BCG_H