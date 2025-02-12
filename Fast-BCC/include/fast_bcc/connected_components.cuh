#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include "extern_bcc/extern_bcc.cuh"       
#include "extern_bcc/bcg_memory_utils.cuh"

void CC(GPU_BCG& g_bcg_ds, const bool isLastBatch);

#endif //CONNECTED_COMPONENTS_H
