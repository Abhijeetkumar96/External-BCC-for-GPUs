#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <unordered_map>
#include <functional>

// Define the hash function for a set
std::size_t hash_set(const std::set<int>& s) {
    std::size_t seed = s.size();
    for (int i : s) {
        seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

// Precompute hash values for a map of sets
std::unordered_map<std::size_t, std::set<int>> precompute_hashes(const std::map<int, std::set<int>>& map) {
    std::unordered_map<std::size_t, std::set<int>> hash_map;
    for (const auto& pair : map) {
        std::size_t hash_value = hash_set(pair.second);
        hash_map[hash_value] = pair.second;
    }
    return hash_map;
}

// Check if all sets in serial_map are available in hashed_par_map
bool areAllSetsAvailable(const std::map<int, std::set<int>>& serial_map, const std::unordered_map<std::size_t, std::set<int>>& hashed_par_map) {
    for (const auto& pair : serial_map) {
        std::size_t hash_value = hash_set(pair.second);

        auto found = hashed_par_map.find(hash_value);
        if (found == hashed_par_map.end() || found->second != pair.second) {
            return false;
        }
    }
    return true;
}

class bcc_result {
public:
 std::map<int, std::set<int>> bcc_validator;
 std::vector<int> cut_vertex;
 void print_results(const std::map<int, std::set<int>>&);
 void print_results(const std::vector<int>&);
 void print() {
     print_results(bcc_validator);
     print_results(cut_vertex);
 }
};

bool read_input(const std::string& filename, std::map<int, std::set<int>>& bcc_validator, std::vector<int>& cut_vertex) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Unable to open file." << std::endl;
        return false;
    }

    int n, e;
    in >> n >> e;
    cut_vertex.resize(n);

    std::string line;
    getline(in, line); // Read the remaining part of the line after integers

    // Read cut vertex status
    do {
        getline(in, line);
    } while (line.empty() && !in.eof());  // Continue until a non-empty line or end of file

    if (line != "cut vertex status") {
        std::cerr << "Expected 'cut vertex status', got: " << line << std::endl;
        return false;
    }

    int u, cut_status;
    for (int i = 0; i < n; ++i) {
        in >> u >> cut_status;
        cut_vertex[u] = cut_status;
    }

    // Read vertex BCC number
    do {
        getline(in, line);
    } while (line.empty() && !in.eof());  // Continue until a non-empty line or end of file

    if (line != "vertex BCC number") {
        std::cerr << "Expected 'vertex BCC number', got: " << line << std::endl;
        return false;
    }

    int bcc_no;
    for (int i = 0; i < n; ++i) {
        in >> u >> bcc_no;
        if (!cut_vertex[u]) {
            bcc_validator[bcc_no].insert(u);
        }
    }
    return true;
}

bool validate(const bcc_result& serial_results, const bcc_result& par_results) {
    // bool isCutVertexSame = serial_results.cut_vertex == par_results.cut_vertex;
    bool isCutVertexSame = true;
    for (int i = 0; i < serial_results.cut_vertex.size(); ++i) {
        if (serial_results.cut_vertex[i] != par_results.cut_vertex[i]) {
            std::cerr << "cut vertex status is different for vertex " << i << "\n";
            isCutVertexSame = false;
        }
    }
    
    if(!isCutVertexSame) {
        std::cout << "Cut Vertex Validation: Failed" << std::endl;
        return false;  // Early exit if cut vertices differ
    }
    
    std::cout << "Cut Vertex Validation: Passed" << std::endl;

    auto hashed_par_map = precompute_hashes(par_results.bcc_validator);
    bool isBccValidatorSame = areAllSetsAvailable(serial_results.bcc_validator, hashed_par_map);

    std::cout << "BCC Validator Validation: " << (isBccValidatorSame ? "Passed" : "Failed") << std::endl;
    
    return isCutVertexSame && isBccValidatorSame;
}


int main(int argc, char* argv[]) {

 if(argc < 3) {
     std::cerr <<"Usage: " << argv[0] <<" <serial_output> <par_output>" << std::endl;
     return EXIT_FAILURE;
 }
 bcc_result serial_results, par_results;
 std::string serial_filename = argv[1];
 std::string par_filename = argv[2];

    bool err;
 err = read_input(serial_filename, serial_results.bcc_validator, serial_results.cut_vertex);
    if(!err) 
        return EXIT_FAILURE;

 err = read_input(par_filename,  par_results.bcc_validator,    par_results.cut_vertex);
    if(!err) 
        return EXIT_FAILURE;

 bool validationSuccess = validate(serial_results, par_results);
    
    if (validationSuccess) {
        std::cout << "Overall Validation: Successful" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "Overall Validation: Failed" << std::endl;
        return EXIT_FAILURE;
    }

}



// #include <algorithm>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <set>
// #include <map>

// #include <omp.h>

// // #define DEBUG

// class bcc_result {
// public:
// 	std::map<int, std::set<int>> bcc_validator;
// 	std::vector<int> cut_vertex;
// 	void print_results(const std::map<int, std::set<int>>&);
// 	void print_results(const std::vector<int>&);
// 	void print() {
// 		print_results(bcc_validator);
// 		print_results(cut_vertex);
// 	}
// };

// bool read_input(const std::string& filename, std::map<int, std::set<int>>& bcc_validator, std::vector<int>& cut_vertex) {
//     std::ifstream in(filename);
//     if (!in) {
//         std::cerr << "Unable to open file." << std::endl;
//         return false;
//     }

//     int n, e;
//     in >> n >> e;
//     cut_vertex.resize(n);

//     std::string line;
//     getline(in, line); // Read the remaining part of the line after integers

//     // Read cut vertex status
//     do {
//         getline(in, line);
//     } while (line.empty() && !in.eof());  // Continue until a non-empty line or end of file

//     if (line != "cut vertex status") {
//         std::cerr << "Expected 'cut vertex status', got: " << line << std::endl;
//         return false;
//     }

//     int u, cut_status;
//     for (int i = 0; i < n; ++i) {
//         in >> u >> cut_status;
//         cut_vertex[u] = cut_status;
//     }

//     // Read vertex BCC number
//     do {
//         getline(in, line);
//     } while (line.empty() && !in.eof());  // Continue until a non-empty line or end of file

//     if (line != "vertex BCC number") {
//         std::cerr << "Expected 'vertex BCC number', got: " << line << std::endl;
//         return false;
//     }

//     int bcc_no;
//     for (int i = 0; i < n; ++i) {
//         in >> u >> bcc_no;
//         if (!cut_vertex[u]) {
//             bcc_validator[bcc_no].insert(u);
//         }
//     }
//     return true;
// }

// void bcc_result::print_results(const std::map<int, std::set<int>>& bcc_validator) {
//     for (const auto& pair : bcc_validator) {
//         std::cout << pair.first << " : ";
//         for (const int vertex : pair.second) {
//             std::cout << vertex << " ";
//         }
//         std::cout << std::endl;
//     }
// }

// void bcc_result::print_results(const std::vector<int>& cut_vertex) {
// 	int ii = 0;
// 	for(const auto &i : cut_vertex) {
// 		std::cout << ii++ <<" : " << i << std::endl;
// 	}
// }

// bool areAllSetsAvailable(const std::map<int, std::set<int>>& map1, const std::map<int, std::set<int>>& map2) {
//     bool allSetsAvailable = true;

//     #pragma omp parallel for shared(allSetsAvailable)
//     for (const auto& pair : map1) {
//         const std::set<int>& set1 = pair.second;
//         bool setFound = false;

//         for (const auto& pair2 : map2) {
//             const std::set<int>& set2 = pair2.second;

//             if (set1 == set2) {
//                 setFound = true;
//                 break;
//             }
//         }

//         if (!setFound) {
//             allSetsAvailable = false;
//         }
//     }
//     return allSetsAvailable;
// }

// bool validate(const bcc_result& serial_results, const bcc_result& par_results) {
//     bool isCutVertexSame = serial_results.cut_vertex == par_results.cut_vertex;

//     #ifdef DEBUG
//         if(!isCutVertexSame) {
//             // for(int i = 0; i < serial_results.cut_vertex.size(); ++i) {
//             //     if(serial_results.cut_vertex[i] != par_results.cut_vertex[i]) {
//             //         std::cout << "serial: "   << i <<" <- " << serial_results.cut_vertex[i]    << std::endl;
//             //         std::cout << "parallel: " << i <<" <- " << par_results.cut_vertex[i]       << std::endl;
//                 // }
//             // }
//         }
//         std::cout << "Cut Vertex Validation: " << (isCutVertexSame ? "Passed" : "Failed") << std::endl;
//     #endif
//     bool isBccValidatorSame = areAllSetsAvailable(serial_results.bcc_validator, par_results.bcc_validator);
//     #ifdef DEBUG
//         std::cout << "BCC Validator Validation: " << (isBccValidatorSame ? "Passed" : "Failed") << std::endl;
//     #endif
    
//     return isCutVertexSame && isBccValidatorSame;
// }

// int main(int argc, char* argv[]) {

// 	if(argc < 3) {
// 		std::cerr <<"Usage: " << argv[0] <<" <serial_output> <par_output>" << std::endl;
// 		return EXIT_FAILURE;
// 	}
// 	bcc_result serial_results, par_results;
// 	std::string serial_filename = argv[1];
// 	std::string par_filename = argv[2];

//     bool err;
// 	err = read_input(serial_filename, serial_results.bcc_validator, serial_results.cut_vertex);
//     if(!err) 
//         return EXIT_FAILURE;

// 	err = read_input(par_filename, 	par_results.bcc_validator, 	  par_results.cut_vertex);
//     if(!err) 
//         return EXIT_FAILURE;

// 	bool validationSuccess = validate(serial_results, par_results);
	
// 	#ifdef DEBUG
// 		// std::cout << "serial_results:\n"; serial_results.print();
// 		// std::cout << "par_results: \n"; par_results.print();
// 	#endif
	
//     if (validationSuccess) {
//         std::cout << "Overall Validation: Successful" << std::endl;
//         return EXIT_SUCCESS;
//     } else {
//         std::cout << "Overall Validation: Failed" << std::endl;
//         return EXIT_FAILURE;
//     }

// }