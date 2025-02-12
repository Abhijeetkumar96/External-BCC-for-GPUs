# Description
- To generate a graph using the tools you've listed, it depends on what type of graph you want to create and the specific task you are trying to accomplish. Here's a brief overview of some of the tools you've mentioned that can be used for graph generation:

* agmgen: Likely used for generating graphs based on the Affiliation Graph Model (AGM). Useful if you're dealing with bipartite graphs or networks where nodes belong to communities or groups.

* graphgen: This is a general-purpose tool for graph generation. If you're looking to create a standard graph with custom specifications (like number of nodes, edges, etc.), this would be a good choice.

* forestfire: This tool probably implements the Forest Fire model for generating graphs. It's used to create networks that mimic certain real-world network properties like the small-world phenomenon.

* kronem, kronfit, krongen: These are related to the Kronecker graph generation method. Kronecker graphs are useful for modeling large-scale networks (like social networks or web graphs) that exhibit properties such as heavy tails in degree distribution.

* randwalk: This might be used for generating graphs based on a random walk process. It's useful for simulating how information or influence might spread through a network.

* temporalmotifsmain: If you're interested in generating graphs that change over time (temporal graphs) and focus on recurring patterns (motifs), this tool could be relevant.

* maggen, magfit: These might be related to the Microsoft Academic Graph generation and fitting. Useful for generating academic citation networks or similar types of graphs.
# Usage
```shell
./graphgen -g:w -n:1000 -k:4 -p:0.1 -o:smallworld.txt
```
will generate a Watts-Strogatz small-world graph where each node is connected to k=4 closest nodes and with the rewiring probability of p=0.1.
```shell
./forestfire -o:myforestfiregraph.txt -n:20 -f:0.2 -b:0.1 -s:1 -a:0.5 -op:0.05
```
- Explanation of the parameters:

	+ -o:myforestfiregraph.txt: This specifies the output file name where the generated graph will be saved. In this 	case, the file will be named "myforestfiregraph.txt".
	+ -n:1000: Sets the number of nodes in the graph to 1000.
	- -f:0.2: Sets the forward burning probability to 0.2. A lower value here tends to create fewer edges and can lead to longer paths.
	+ -b:0.1: Sets the backward burning probability to 0.1, which is even lower than the forward probability, again 	favoring the creation of longer paths.
	+ -s:1: Starts the graph with 1 isolated node.
	+ -a:0.5: Sets the probability of a new node choosing 2 ambassadors to 0.5. This parameter controls how new nodes 	connect to the existing network.
	+ -op:0.05: Sets the probability of a new node being an orphan (a node with zero out-degree) to 0.05.
	
Remember, the Forest Fire model is stochastic, meaning that it involves randomness. Therefore, the exact structure of the generated graph can vary with each run, even with the same parameters. The parameters should be adjusted based on the specific characteristics you want in your graph. 

# ref: 
1. https://snap.stanford.edu/snap/description.html
2. https://snap.stanford.edu/snap-1.8/doc.html
