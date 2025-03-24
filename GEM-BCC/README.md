# External_BCC Project Structure

# Directory Structure

```plaintext
/External_BCC
  ├── CMakeLists.txt                    # Root CMake configuration file
  ├── README.md                         # Project description and instructions
  ├── datasets                          # Dataset files for testing and usage
  ├── include                           # Header files directory
  │   ├── cuda_bcc                      # CUDA implementations for BCC algorithms
  │   │   ├── bcc.cuh
  │   │   ├── bfs.cuh
  │   │   ├── connected_components.cuh
  │   │   ├── cuda_csr.cuh
  │   │   ├── cut_vertex.cuh
  │   │   └── lca.cuh
  │   ├── extern_bcc                    # BCG related data structures
  │   │   ├── bcg.cuh
  │   │   ├── bcg_memory_utils.cuh
  │   │   ├── edge_cleanup.cuh
  │   │   ├── extern_bcc.cuh
  │   │   └── repair.cuh
  │   ├── graph                         # Graph data structure and operations
  │   │   └── graph.cuh
  │   ├── multicore_bfs                 # Multicore BFS algorithm headers
  │   │   ├── multicore_bfs.cuh
  │   │   └── thread.hpp
  │   └── utility                       # Utility functions for CUDA and general use
  │       ├── CommandLineParser.cuh
  │       ├── cuda_utility.cuh
  │       ├── export_output.cuh
  │       ├── timer.hpp
  │       └── utility.hpp
  ├── src                               # Source files directory
  │   ├── cuda_bcc                      # Source implementations for CUDA BCC
  │   │   ├── bcc.cu
  │   │   ├── bfs.cu
  │   │   ├── checker_v1.cpp
  │   │   ├── connected_components.cu
  │   │   ├── cuda_csr.cu
  │   │   ├── cut_vertex.cu
  │   │   ├── implicit_bcc_validator.cpp
  │   │   └── lca.cu
  │   ├── extern_bcc                    # Source implementations for external BCC
  │   │   ├── bcg.cu
  │   │   ├── bcg_memory_utils.cu
  │   │   ├── edge_cleanup.cu
  │   │   ├── extern_bcc.cu
  │   │   └── repair.cu
  │   ├── graph                         # Graph-related source implementations
  │   │   └── graph.cpp
  │   ├── main.cu                       # Main program source file for CUDA
  │   └── multicore_bfs                 # Source implementation for multicore BFS
  │       └── multicore_bfs.cpp
  └── tools                             # Auxiliary tools and scripts for graph processing
      ├── Makefile
      ├── SNAP
      │   ├── snap
      │   │   └── README.md
      │   └── snap_graph_parser.cpp
      ├── bin
      │   └── Serial_BCC
      ├── ecl_runner.sh
      └── src
          ├── ECLgraph.h
          ├── Serial_BCC_v1.cpp
          ├── edge_list_to_ecl.cpp
          ├── edge_mirror.cpp
          ├── snap_graph_parser.cpp
          └── snap_to_txt.cpp
```

# Building Your Project
Follow the same steps as before:

- Create a build directory and navigate into it:
```shell
mkdir build && cd build
```
- Run CMake to generate the build system:
```shell
cmake -DCMAKE_BUILD_TYPE=Release ..
```

- or Run CMake to generate the DEBUG system:
```shell
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

- Compile your project:
```shell
make -j$(nproc)
```
- To enable verbose output for troubleshooting build issues:
```shell
make VERBOSE=1 -j$(nproc)
```