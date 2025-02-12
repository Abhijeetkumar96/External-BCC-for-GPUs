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
#include "utility/export_output.cuh"     
#include "utility/CommandLineParser.cuh" 

//---------------------------------------------------------------------
// BFS and Biconnected Components (BCC) related functions
//---------------------------------------------------------------------
#include "multicore_bfs/multicore_bfs.hpp" 
#include "extern_bcc/bcg.cuh"              
#include "extern_bcc/extern_bcc.cuh"       
#include "extern_bcc/bcg_memory_utils.cuh" 


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    checker             = false;
bool    g_verbose           = false;  // Whether to display i/o to console
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
            std::cout << "\n\nOutputting CSR representation: \n";
            g.print_CSR();

            std::cout <<"\n\nEdgeList:\n";
            g.print_edgelist();
        }
        // Create External - BCC object
        GPU_BCG gpu_bcg(num_vert, num_edges, args.batchSize);
        
        // Create BFS object 
        BFS bfs(g.getEdges().data(), g.getVertices().data(),
                g.getParent(), g.getLevel(),
                g.getNumVertices(), g.getNumEdges() / 2, 
                g.getAvgOutDegree(), g.getMaxDegreeVert());
        
        Timer myTimer;
        // start external_bcc
        extern_bcc(g, bfs, gpu_bcg);

        auto dur = myTimer.stop();

        // Setup output for results
        bool write_output = args.write_output;
        std::string output_path = args.output_directory;

        myTimer.reset();
        
        // write_output = true;

        // Initialize Output Manager
        BCCOutputManager bcc_mag(
            num_vert, num_edges, 
            g.getSrc(), g.getDest(), 
            gpu_bcg, get_file_extension(filename), 
            output_path, dur);
        
        if(write_output) {
            // Write results to file
            bcc_mag.write(gpu_bcg);
            dur = myTimer.stop();
            std::cout <<"\nWriting output finished in " << formatDuration(dur) << "." << std::endl;
        } else {
            // Display summary to terminal
            bcc_mag.show_summary(gpu_bcg);
            dur = myTimer.stop();
            std::cout << "\nSummary displayed in " << formatDuration(dur) << "." << std::endl;
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// ====[ End of Main Code ]====