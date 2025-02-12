CXX=g++
CXXFLAGS=-std=c++17 -O3 -Wall

SRCDIR := src
INCDIR := include
BINDIR := bin

$(shell mkdir -p $(BINDIR))

all: $(BINDIR)/snap_graph_parser $(BINDIR)/edge_list_to_ecl $(BINDIR)/snap_to_txt $(BINDIR)/edge_mirror $(BINDIR)/Serial_BCC_v1

$(BINDIR)/snap_graph_parser: $(SRCDIR)/snap_graph_parser.cpp
	$(CXX) $(CXXFLAGS) $(SRCDIR)/snap_graph_parser.cpp -o $(BINDIR)/snap_graph_parser

$(BINDIR)/edge_list_to_ecl: $(SRCDIR)/edge_list_to_ecl.cpp
	$(CXX) $(CXXFLAGS) $(SRCDIR)/edge_list_to_ecl.cpp -o $(BINDIR)/edge_list_to_ecl

$(BINDIR)/snap_to_txt: $(SRCDIR)/snap_to_txt.cpp
	$(CXX) $(CXXFLAGS) $(SRCDIR)/snap_to_txt.cpp -o $(BINDIR)/snap_to_txt

$(BINDIR)/edge_mirror: $(SRCDIR)/edge_mirror.cpp
	$(CXX) $(CXXFLAGS) $(SRCDIR)/edge_mirror.cpp -o $(BINDIR)/edge_mirror

$(BINDIR)/Serial_BCC_v1: $(SRCDIR)/Serial_BCC_v1.cpp
	$(CXX) $(CXXFLAGS) $(SRCDIR)/Serial_BCC_v1.cpp -o $(BINDIR)/Serial_BCC_v1

clean:
	rm $(BINDIR)/*

.PHONY: all clean