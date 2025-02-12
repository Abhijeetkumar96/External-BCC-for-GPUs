#ifndef LCA_H
#define LCA_H

#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include "extern_bcc/bcg_memory_utils.cuh"

void naive_lca(GPU_BCG& g_bcg_ds, int root);

#endif // LCA_H