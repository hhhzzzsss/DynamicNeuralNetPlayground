import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from dnn_libs.agraph import AcyclicGraph, NodeData
import os

import cProfile
import pstats

try:
    profile
except NameError:
    def profile(func): return func

# Set the device to CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Download and load the test data
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Create an acyclic graph using our implementation
G = AcyclicGraph()

# Node mappings for input/output/hidden nodes
node_mappings = {}
reverse_mappings = {}

# Add input nodes (28*28 vertices)
input_nodes = []
for i in range(28*28):
    node_id = G.add_node(hunger=0.0, utility=0.0)
    node_key = ('i', i)
    node_mappings[node_key] = node_id
    reverse_mappings[node_id] = node_key
    input_nodes.append(node_key)

# Add output nodes (10 vertices)
output_nodes = []
for i in range(10):
    node_id = G.add_node(hunger=0.0, utility=0.0)
    node_key = ('o', i)
    node_mappings[node_key] = node_id
    reverse_mappings[node_id] = node_key
    output_nodes.append(node_key)

# Add hidden nodes
hidden_nodes = []
def add_hidden_node():
    hidden_node = ('h', len(hidden_nodes))
    node_id = G.add_node(hunger=0.0, utility=0.0) 
    node_mappings[hidden_node] = node_id
    reverse_mappings[node_id] = hidden_node
    hidden_nodes.append(hidden_node)
    return hidden_node

def add_edge(from_node, to_node, sign=0):
    G.add_edge(node_mappings[from_node], node_mappings[to_node], 0.001*sign)

def add_double_edge(from_node, to_node, sign=0):
    hidden_node = add_hidden_node()
    G.add_edge(node_mappings[from_node], node_mappings[hidden_node], 1)
    G.add_edge(node_mappings[hidden_node], node_mappings[to_node], 0.001*sign)

# Helper functions for graph operations
def get_node_attr(node, attr):
    node_id = node_mappings[node]
    if attr == 'utility':
        return G.get_node_utility(node_id)
    elif attr == 'hunger':
        return G.get_node_hunger(node_id)
    else:
        raise ValueError(f"Unknown attribute: {attr}")

def set_node_attr(node, attr, value):
    node_id = node_mappings[node]
    if attr == 'utility':
        G.set_node_utility(node_id, value)
    elif attr == 'hunger':
        G.set_node_hunger(node_id, value)
    else:
        raise ValueError(f"Unknown attribute: {attr}")

def get_in_edges(node):
    node_id = node_mappings[node]
    edges = G.get_in_edges(node_id)
    return [(reverse_mappings[edge[0]], node) for edge in edges]

def get_edge_weight(from_node, to_node):
    from_id = node_mappings[from_node]
    to_id = node_mappings[to_node]
    return G.get_edge_weight(from_id, to_id)

def set_edge_weight(from_node, to_node, weight):
    from_id = node_mappings[from_node]
    to_id = node_mappings[to_node]
    G.set_edge_weight(from_id, to_id, weight)

def get_predecessors(node):
    node_id = node_mappings[node]
    edges = G.get_in_edges(node_id)
    return [reverse_mappings[edge[0]] for edge in edges]

def get_descendants(start_node):
    # BFS to find all descendants
    queue = [node_mappings[start_node]]
    visited = set(queue)
    descendants = set()
    
    while queue:
        current = queue.pop(0)
        for edge in G.get_out_edges(current):
            target = edge[0]
            if target not in visited:
                visited.add(target)
                queue.append(target)
                descendants.add(reverse_mappings[target])
                
    return descendants

def get_node_delta_utility(node):
    # We need to store delta_utility separately since it's not in C++ implementation
    if not hasattr(get_node_delta_utility, "values"):
        get_node_delta_utility.values = {n: 0.0 for n in input_nodes + output_nodes + hidden_nodes}
    return get_node_delta_utility.values.get(node, 0.0)

def set_node_delta_utility(node, value):
    if not hasattr(set_node_delta_utility, "values"):
        set_node_delta_utility.values = {n: 0.0 for n in input_nodes + output_nodes + hidden_nodes}
    set_node_delta_utility.values[node] = value

def get_topological_sort():
    # Get sorted node IDs from C++ implementation
    sorted_ids = G.get_topological_order()
    # Convert to node tuples
    return [reverse_mappings[node_id] for node_id in sorted_ids]

def predict_batch(inputs, labels, utility_decay=0.99, hunger_decay=0.95, learning_rate=0.01):
    # Nodes in topological order
    # Input and output nodes are further forced to be at the front and back respectively
    sorted_nodes = get_topological_sort()
    sorted_nodes = input_nodes + [node for node in sorted_nodes if node[0] == 'h'] + output_nodes

    # Dictionary mapping nodes to their index in sorted_nodes
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

    # Edges in order of when they should be processed
    sorted_edges = []
    for node in sorted_nodes:
        sorted_edges.extend(get_in_edges(node))

    # Create a dictionary mapping edges to their indices
    edge_to_index = {edge: idx for idx, edge in enumerate(sorted_edges)}

    # Weight tensor for edges
    edge_weights = torch.tensor([get_edge_weight(edge[0], edge[1]) for edge in sorted_edges], requires_grad=True)
    if (len(edge_weights) == 0):
        edge_weights = torch.tensor([0.0], requires_grad=True)

    # Process each sample in the batch
    edge_growth_intents = {}
    batch_weight_update = torch.zeros_like(edge_weights)
    batch_loss = 0
    for input, label in zip(inputs, labels):
        edge_weights.grad = torch.zeros_like(edge_weights)
        weight_update, loss = predict_sample(input, label, edge_growth_intents, sorted_nodes, node_to_index, sorted_edges, edge_to_index, edge_weights, utility_decay, hunger_decay)
        batch_weight_update += weight_update
        batch_loss += loss

    print(f"Batch weight update: {batch_weight_update}")

    # Update edge weights
    for edge in sorted_edges:
        new_weight = get_edge_weight(edge[0], edge[1]) + learning_rate * batch_weight_update[edge_to_index[edge]]
        set_edge_weight(edge[0], edge[1], new_weight)

    # Get largest edge growth intent across all samples
    if edge_growth_intents:
        largest_intent_edge, intent_value = max(edge_growth_intents.items(), key=lambda x: abs(x[1]))
        from_node = sorted_nodes[largest_intent_edge[0]]
        to_node = sorted_nodes[largest_intent_edge[1]]
        sign = 1 if intent_value > 0 else -1
        if get_node_attr(to_node, 'utility') > 1.0:
            add_double_edge(from_node, to_node, sign)
        else:
            add_edge(from_node, to_node, sign)

    return batch_loss

# Get the predictions for a single sample
@profile
def predict_sample(input, label, edge_growth_intents, sorted_nodes, node_to_index, sorted_edges, edge_to_index, edge_weights, utility_decay=0.99, hunger_decay=0.95):
    # Node value tensors for autograd
    node_values = [None] * len(sorted_nodes)
    for i in range(len(input_nodes)):
        node_values[i] = input[i].clone().detach()

    # Forward pass
    for node in sorted_nodes[len(input_nodes):]:
        incoming_edges = get_in_edges(node)
        if incoming_edges:
            for edge in incoming_edges:
                assert node_values[node_to_index[edge[0]]] is not None
                assert edge_weights[edge_to_index[edge]] is not None
            incoming_products = torch.stack([node_values[node_to_index[edge[0]]] * edge_weights[edge_to_index[edge]] for edge in incoming_edges])
            node_values[node_to_index[node]] = torch.relu(torch.sum(incoming_products)).requires_grad_(True)
            node_values[node_to_index[node]].retain_grad()
        else:
            node_values[node_to_index[node]] = torch.tensor(0.0, requires_grad=True)

    # Stack output node values into a single tensor
    output_tensor = torch.stack([node_values[node_to_index[node]] for node in output_nodes])

    # Calculate cross entropy loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output_tensor.unsqueeze(0), torch.tensor([label]))

    # Backpropagate
    loss.backward()

    # Add 1/n to delta_utility for each output node
    for output_node in output_nodes:
        set_node_delta_utility(output_node, get_node_delta_utility(output_node) + 1.0 / len(output_nodes))

    # Process nodes in reverse order
    for node in reversed(sorted_nodes):
        # Add current delta_utility to utility, and multiply utility by alpha
        current_utility = get_node_attr(node, 'utility')
        current_utility += get_node_delta_utility(node)
        current_utility *= utility_decay
        set_node_attr(node, 'utility', current_utility)
        
        # Propagate utility backwards
        incoming_edges = get_in_edges(node)
        if incoming_edges:
            # Get total weighted input for normalization
            total_weight_value = sum(abs(get_edge_weight(e[0], e[1]) * node_values[node_to_index[e[0]]]) for e in incoming_edges)
            
            # If total_weight_value is not zero, distribute delta_utility
            if total_weight_value != 0:
                for edge in incoming_edges:
                    # Calculate proportion of delta_utility to pass back
                    weight = get_edge_weight(edge[0], edge[1])
                    value = node_values[node_to_index[edge[0]]]
                    proportion = abs(weight * value) / total_weight_value
                    
                    # Add to predecessor's delta_utility
                    pred_delta = get_node_delta_utility(edge[0]) + get_node_delta_utility(node) * proportion
                    set_node_delta_utility(edge[0], pred_delta)
            
        # Reset delta_utility for this node
        for edge in incoming_edges:
            set_node_delta_utility(node, 0.0)

    # Update hunger
    for node in sorted_nodes[len(input_nodes):]:
        current_hunger = get_node_attr(node, 'hunger')
        current_hunger += abs(node_values[node_to_index[node]].grad)
        current_hunger *= hunger_decay
        set_node_attr(node, 'hunger', current_hunger)

    # Calculate weight updates
    weight_update = -edge_weights.grad
    for i, edge in enumerate(sorted_edges):
        weight_update[i] /= (get_node_attr(edge[0], 'utility') + 1.0)

    # Get nodes with at most average in-degree
    non_input_nodes = sorted_nodes[len(input_nodes):]
    in_degrees = [len(get_in_edges(node)) for node in non_input_nodes]
    avg_in_degree = sum(in_degrees) / len(in_degrees) if in_degrees else 0
    below_avg_nodes = [node for node, deg in zip(non_input_nodes, in_degrees) if deg <= avg_in_degree]
    
    # Find node with largest hunger
    max_hunger_node = max(below_avg_nodes, key=lambda n: get_node_attr(n, 'hunger')) if below_avg_nodes else None
    
    if max_hunger_node:
        # Get all nodes that are not descendants or in-neighbors of max_hunger_node
        in_neighbors = set(get_predecessors(max_hunger_node))
        descendants = get_descendants(max_hunger_node)
        all_nodes = set(sorted_nodes)
        potential_in_neighbors = all_nodes - descendants - in_neighbors - {max_hunger_node}

        for potential_in_neighbor in potential_in_neighbors:
            # Calculate edge growth intent
            edge_repr = (node_to_index[potential_in_neighbor], node_to_index[max_hunger_node])
            edge_growth_intents[edge_repr] = edge_growth_intents.get(edge_repr, 0) + node_values[edge_repr[0]] * node_values[edge_repr[1]].grad

    return weight_update, loss

def draw_neural_network(G, ax):
    # Clear the plot
    ax.clear()
    
    # Identify input, hidden, and output nodes
    input_nodes_set = {node_mappings[n] for n in input_nodes}
    output_nodes_set = {node_mappings[n] for n in output_nodes}
    hidden_nodes_set = {node_mappings[n] for n in hidden_nodes}
    
    # Create position dictionary
    pos = {}
    
    # Assign positions for visualization
    # Input nodes in leftmost column
    y_pos = np.linspace(-1, 1, len(input_nodes))
    for i, node in enumerate(input_nodes):
        pos[node_mappings[node]] = (0, -y_pos[i])
        
    # Output nodes in rightmost column
    all_nodes = G.get_nodes()
    max_rank = 2  # Default if there are no hidden nodes
    
    # Determine ranks for hidden nodes (using a simple approach)
    if hidden_nodes:
        topological_order = G.get_topological_order()
        ranks = {node_id: 0 for node_id in input_nodes_set}
        
        # Simple BFS to assign ranks
        for node_id in topological_order:
            if node_id in input_nodes_set:
                continue
                
            max_pred_rank = 0
            for pred_edge in G.get_in_edges(node_id):
                pred_id = pred_edge[0]
                if pred_id in ranks:
                    max_pred_rank = max(max_pred_rank, ranks[pred_id])
            
            ranks[node_id] = max_pred_rank + 1
            max_rank = max(max_rank, ranks[node_id])
    
    # Position output nodes at the rightmost rank
    y_pos = np.linspace(-1, 1, len(output_nodes))
    for i, node in enumerate(output_nodes):
        pos[node_mappings[node]] = (max_rank + 1, -y_pos[i])
    
    # Position hidden nodes based on their rank
    if hidden_nodes:
        # Group hidden nodes by rank
        rank_to_nodes = {}
        for node in hidden_nodes:
            node_id = node_mappings[node]
            r = ranks.get(node_id, 1)  # Default to rank 1 if not in ranks
            if r not in rank_to_nodes:
                rank_to_nodes[r] = []
            rank_to_nodes[r].append(node_id)
        
        # Position nodes within each rank
        for r, nodes_in_rank in rank_to_nodes.items():
            y_pos = np.linspace(-1, 1, len(nodes_in_rank))
            for i, node_id in enumerate(nodes_in_rank):
                pos[node_id] = (r, -y_pos[i])
    
    # Build a color map for each node
    color_map = []
    node_sizes = []
    for node_id in all_nodes:
        if node_id in input_nodes_set:
            color_map.append('lightblue')
            node_sizes.append(50)
        elif node_id in output_nodes_set:
            color_map.append('lightgreen')
            node_sizes.append(50)
        else:
            color_map.append('lightgray')
            node_sizes.append(30)  # Hidden nodes slightly smaller
    
    # Create edge list and weights for drawing
    edges = []
    edge_weights = []
    
    for node_id in all_nodes:
        for edge in G.get_out_edges(node_id):
            edges.append((node_id, edge[0]))
            edge_weights.append(abs(edge[1]) * 3)  # Scale weights for visibility
    
    # Draw the nodes
    nx.draw_networkx_nodes(
        nx.DiGraph(), 
        pos, 
        nodelist=all_nodes, 
        node_color=color_map,
        node_size=node_sizes
    )
    
    # Draw the edges
    nx.draw_networkx_edges(
        nx.DiGraph(), 
        pos, 
        edgelist=edges, 
        width=edge_weights,
        edge_color="gray",
        arrows=True,
        arrowsize=10,
        arrowstyle='->'
    )
    
    plt.pause(0.01)

def main():
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    # Get a batch from the trainloader and print it
    # limit = 20
    for batch_num, (inputs, labels) in enumerate(trainloader):
        print(inputs.shape, labels.shape)
        # loss = predict_batch(inputs.reshape(32, -1), labels)
        # print(f"Batch {batch_num}, {loss=}")
        # draw_neural_network(G, ax)

        # limit -= 1
        # if limit <= 0:
        #     break

# Re-add networkx import just for visualization functions
import networkx as nx
main()