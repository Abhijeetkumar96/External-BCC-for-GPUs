import pydot

def visualize_tree(parent):
    print("Starting tree visualization...")
    # Create a graph object
    graph = pydot.Dot(graph_type='graph')

    print("Adding edges...")
    for child_index in range(len(parent)):
        parent_index = parent[child_index]
        # For non-root nodes, add an edge from parent to child
        if parent_index != child_index:  # Assuming root points to itself
            graph.add_edge(pydot.Edge(str(parent_index), str(child_index)))
            print(f"Added edge from {parent_index} to {child_index}")

    # Save and visualize the graph
    graph.write_png('tree_visualization.png')
    print("Tree visualization completed and saved to 'tree_visualization.png'.")

def read_parent_array_from_file(file_path):
    print("Reading parent array from file...")
    with open(file_path, 'r') as file:
        # Read the first line and convert it to a list of integers
        parent = list(map(int, file.readline().strip().split(',')))
    print("Parent array successfully read.")
    return parent

# File path
file_path = 'parent_array.txt'

# Read parent array from the file
parent = read_parent_array_from_file(file_path)

# Visualize the tree
visualize_tree(parent)
