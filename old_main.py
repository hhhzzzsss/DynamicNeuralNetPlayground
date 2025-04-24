import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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

# Create an empty directed acyclic graph
G = nx.DiGraph()

# Add input nodes (28*28 vertices)
input_nodes = [('i', i) for i in range(28*28)]
G.add_nodes_from(input_nodes, utility=0.0, delta_utility=0.0, hunger=0.0)

# Add output nodes (10 vertices)
output_nodes = [('o', i) for i in range(10)]
G.add_nodes_from(output_nodes, utility=0.0, delta_utility=0.0, hunger=0.0)

# # Add initial edges from each input node to each output node
# for input_node in input_nodes:
#     for output_node in output_nodes:
#         G.add_edge(input_node, output_node, weight=0.001)

# Add hidden nodes
hidden_nodes = []
def add_hidden_node():
    hidden_node = ('h', len(hidden_nodes))
    hidden_nodes.append(hidden_node)
    G.add_node(hidden_node, utility=0.0, delta_utility=0.0, hunger=0.0)
    return hidden_node

def add_edge(from_node, to_node, sign=0):
    G.add_edge(from_node, to_node, weight=0.001*sign)

def add_double_edge(from_node, to_node, sign=0):
    hidden_node = add_hidden_node()
    G.add_edge(from_node, hidden_node, weight=1)
    G.add_edge(hidden_node, to_node, weight=0.001*sign)

def predict_batch(inputs, labels, utility_decay=0.99, hunger_decay=0.95, learning_rate=0.01):
    # Nodes in topological order
    # Input and output nodes are further forced to be at the front and back respectively
    sorted_nodes = list(nx.topological_sort(G))
    sorted_nodes = input_nodes + [node for node in sorted_nodes if node[0] == 'h'] + output_nodes

    # Dictionary mapping nodes to their index in sorted_nodes
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

    # Edges in order of when they should be processed
    sorted_edges = []
    for node in sorted_nodes:
        sorted_edges.extend(G.in_edges(node))

    # Create a dictionary mapping edges to their indices
    edge_to_index = {edge: idx for idx, edge in enumerate(sorted_edges)}

    # Weight tensor for edges
    edge_weights = torch.tensor([G.get_edge_data(*edge)['weight'] for edge in sorted_edges], requires_grad=True)
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
        G[edge[0]][edge[1]]['weight'] += learning_rate * batch_weight_update[edge_to_index[edge]]

    # Get largest edge growth intent across all samples
    if edge_growth_intents:
        largest_intent_edge, intent_value = max(edge_growth_intents.items(), key=lambda x: abs(x[1]))
        from_node = sorted_nodes[largest_intent_edge[0]]
        to_node = sorted_nodes[largest_intent_edge[1]]
        sign = 1 if intent_value > 0 else -1
        if G.nodes[to_node]['utility'] > 1.0:
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
        incoming_edges = list(G.in_edges(node))
        if incoming_edges:
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

    # Print logits, prediction, and label
    # print(f"Logits: {output_tensor}, Prediction: {torch.argmax(output_tensor).item()}, Label: {label}")



    # Add 1/n to delta_utility for each output node
    for output_node in output_nodes:
        G.nodes[output_node]['delta_utility'] += 1.0 / len(output_nodes)

    # Process nodes in reverse order
    for node in reversed(sorted_nodes):
        # Add current delta_utility to utility, and multiply utility by alpha
        G.nodes[node]['utility'] += G.nodes[node]['delta_utility']
        G.nodes[node]['utility'] *= utility_decay
        
        # Propagate utility backwards
        incoming_edges = list(G.in_edges(node))
        if incoming_edges:
            # Get total weighted input for normalization
            total_weight_value = sum(abs(G[e[0]][e[1]]['weight'] * node_values[node_to_index[e[0]]]) for e in incoming_edges)
            
            # If total_weight_value is not zero, distribute delta_utility
            if total_weight_value != 0:
                for edge in incoming_edges:
                    # Calculate proportion of delta_utility to pass back
                    weight = G[edge[0]][edge[1]]['weight']
                    value = node_values[node_to_index[edge[0]]]
                    proportion = abs(weight * value) / total_weight_value
                    
                    # Add to predecessor's delta_utility
                    G.nodes[edge[0]]['delta_utility'] += G.nodes[edge[1]]['delta_utility'] * proportion
            
        # Reset delta_utility for this node
        for edge in incoming_edges:
            G.nodes[node]['delta_utility'] = 0.0



    # Update hunger
    for node in sorted_nodes[len(input_nodes):]:
        G.nodes[node]['hunger'] += abs(node_values[node_to_index[node]].grad)
        G.nodes[node]['hunger'] *= hunger_decay



    # Calculate weight updates
    weight_update = -edge_weights.grad
    for i, edge in enumerate(sorted_edges):
        weight_update[i] /= (G.nodes[edge[0]]['utility'] + 1.0)



    # Get nodes with at most average in-degree
    non_input_nodes = sorted_nodes[len(input_nodes):]
    in_degrees = [G.in_degree(node) for node in non_input_nodes]
    avg_in_degree = sum(in_degrees) / len(in_degrees)
    below_avg_nodes = [node for node, deg in zip(non_input_nodes, in_degrees) if deg <= avg_in_degree]
    
    # Find node with largest hunger
    max_hunger_node = max(below_avg_nodes, key=lambda n: G.nodes[n]['hunger'])

    # Get all nodes that are not descendants or in-neighbors of max_hunger_node
    in_neighbors = set(G.predecessors(max_hunger_node))
    descendants = nx.descendants(G, max_hunger_node)
    potential_in_neighbors = set(G.nodes()) - descendants - in_neighbors - {max_hunger_node}

    for potential_in_neighbor in potential_in_neighbors:
        # Calculate edge growth intent
        edge_repr = (node_to_index[potential_in_neighbor], node_to_index[max_hunger_node])
        edge_growth_intents[edge_repr] = edge_growth_intents.get(edge_repr, 0) + node_values[edge_repr[0]] * node_values[edge_repr[1]].grad

    return weight_update, loss

def draw_neural_network(G, ax):
    # Identify input and output nodes based on their tuple structure
    input_nodes = {n for n in G.nodes if n[0] == 'i'}
    output_nodes = {n for n in G.nodes if n[0] == 'o'}

    # Assign ranks using a longest-path BFS-like algorithm
    rank = {n: 0 for n in input_nodes}  # Start inputs at rank 0
    visited = set(input_nodes)
    queue = list(input_nodes)

    while queue:
        node = queue.pop(0)
        for neighbor in G.successors(node):
            if neighbor not in visited:
                rank[neighbor] = rank[node] + 1
                queue.append(neighbor)
                visited.add(neighbor)
            else:
                rank[neighbor] = max(rank[neighbor], rank[node] + 1)

    # Ensure output nodes are in the highest rank
    max_rank = max((r for n, r in rank.items() if n[0] in ['i', 'h']), default=0)
    for node in output_nodes:
        rank[node] = max_rank + 1

    # Group nodes by rank
    rank_to_nodes = {}
    for node, r in rank.items():
        rank_to_nodes.setdefault(r, []).append(node)

    # Assign positions for visualization (x = rank, y = spread among that rank)
    pos = {}
    for r, nodes in rank_to_nodes.items():
        nodes = sorted(nodes, key=lambda x: x[1])  # Sort nodes by their index within their type
        y_pos = np.linspace(-1, 1, len(nodes))  # Spread nodes vertically
        for i, node in enumerate(nodes):
            pos[node] = (r, -y_pos[i])  # Negative y so ranks go top->down in matplotlib

    # Build a color map for each node
    #  i -> lightblue, o -> lightgreen, h -> lightgray
    color_map = []
    for node in G.nodes:
        if isinstance(node, tuple):
            if node[0] == 'i':
                color_map.append('lightblue')
            elif node[0] == 'o':
                color_map.append('lightgreen')
            else:
                # treat all other tuples (likely 'h') as hidden
                color_map.append('lightgray')
        else:
            # For any non-tuple nodes (if any),
            # default to hidden color or something else
            color_map.append('lightgray')

    # Draw the graph
    ax.clear()
    nx.draw(
        G,
        pos,
        ax,
        with_labels=False,
        node_color=color_map,
        edge_color="gray",
        node_size=50,
        font_size=8
    )
    plt.pause(0.01)

def main():
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    # Get a batch from the trainloader and print it
    limit = 20
    for batch_num, (inputs, labels) in enumerate(trainloader):
        loss = predict_batch(inputs.reshape(32, -1), labels)
        print(f"Batch {batch_num}, {loss=}")
        draw_neural_network(G, ax)

        limit -= 1
        if limit <= 0:
            break

# cProfile.run('main()', 'main.prof')
# stats = pstats.Stats('main.prof')
# stats.sort_stats('cumulative').print_stats(20)
main()