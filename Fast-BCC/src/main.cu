//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <iostream>  
#include <random> 

//---------------------------------------------------------------------
// Graph data-structure
//---------------------------------------------------------------------
#include "graph/graph.cuh" 

//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "utility/timer.hpp"             
#include "utility/utility.hpp"           
#include "utility/cuda_utility.cuh"
#include "utility/CommandLineParser.cuh" 

//---------------------------------------------------------------------
// Biconnected Components (BCC) related functions
//---------------------------------------------------------------------
#include "extern_bcc/extern_bcc.cuh"       
#include "extern_bcc/bcg_memory_utils.cuh"


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    checker             = false;
bool    g_verbose           = false;  // Whether to display i/o to console
int     local_block_size    = 1000;
long    maxThreadsPerBlock  = 0;

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);

    // Initialize command line parser and retrieve parsed arguments.
    CommandLineParser cmdParser(argc, argv);
    const auto& args = cmdParser.getArgs();
        
    if (args.error) {
        std::cerr << CommandLineParser::help_msg << std::endl;
        exit(EXIT_FAILURE);
    }

    try {
        std::string filename = args.inputFile;
        std::cout <<"\nReading " << get_file_extension(filename) << " file." << std::endl << std::endl;
        undirected_graph g(filename);

        int     num_vert    =   g.getNumVertices();
        long    num_edges   =   g.getNumEdges() / 2;
        
        // Initialize device
        cuda_init(args.cudaDevice);
        g_verbose = args.verbose;
        checker   = args.g_checker_mode;

        // print basic_stats
        g.basic_stats(maxThreadsPerBlock, g_verbose, checker);

        if(g_verbose) {
            std::cout <<"\n\nEdgeList:\n";
            g.print_edgelist();
        }
        // Create External - BCC object
        GPU_BCG gpu_bcg(num_vert, num_edges, args.batchSize);

        gpu_bcg.h_edgelist = g.getList();

        std::cout << "Starting External BCC:" << std::endl;
        
        // start external_bcc
        int bcc_count = extern_bcc(gpu_bcg);

        std::cout << "The number of Biconnected Components (BCCs) in " << get_file_extension(filename) << " is: " << bcc_count << "\n";


    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// ====[ End of Main Code ]====