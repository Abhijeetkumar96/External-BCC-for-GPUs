# Define compiler, flags, and directories
CXX=g++
CXXFLAGS=-std=c++17 -O3
SRC_DIR=src
BIN_DIR=bin

# Targets
all: directories $(BIN_DIR)/EdgeAdderForConnectivity $(BIN_DIR)/edge_list_to_bin $(BIN_DIR)/edge_list_validator $(BIN_DIR)/random_graph_generator

directories:
	@echo "Creating directories..."
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/EdgeAdderForConnectivity: $(SRC_DIR)/EdgeAdderForConnectivity.cpp
	@echo "Compiling EdgeAdderForConnectivity..."
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/edge_list_to_bin: $(SRC_DIR)/edge_list_to_bin.cpp
	@echo "Compiling edge_list_to_bin..."
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/edge_list_validator: $(SRC_DIR)/edge_list_validator.cpp
	@echo "Compiling edge_list_validator..."
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/random_graph_generator: $(SRC_DIR)/random_graph_generator.cpp
	@echo "Compiling random_graph_generator..."
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	@echo "Cleaning up..."
	@rm -rf $(BIN_DIR)

.PHONY: all clean directories
