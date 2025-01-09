import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import networkx as nx
import os

# Set the device to CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Create an empty directed acyclic graph
G = nx.DiGraph()

# Add input nodes (28*28 vertices)
input_nodes = [('i', i) for i in range(28*28)]
G.add_nodes_from(input_nodes, utility=0.0, delta_utility=0.0, hunger=0.0)

# Add output nodes (10 vertices)
output_nodes = [('o', i) for i in range(10)]
G.add_nodes_from(output_nodes, utility=0.0, delta_utility=0.0, hunger=0.0)

# Add hidden nodes
hidden_nodes = []
def add_hidden_node():
    hidden_node = ('h', len(hidden_nodes))
    hidden_nodes.append(hidden_node)
    G.add_node(hidden_node, utility=0.0, delta_utility=0.0, hunger=0.0)

def add_edge(from_node, to_node):
    G.add_edge(from_node, to_node, weight=0.0)

add_edge(('i', 0), ('o', 5))

edge_growth_intents = {}

# Get the predictions for a single sample
def predict_sample(input, label, utility_decay=0.99, hunger_decay=0.95):
    # Get topologically sorted nodes for processing
    sorted_nodes = list(nx.topological_sort(G))
    # Force the input nodes to be at the front and the output nodes to be at the back
    sorted_nodes = input_nodes + [node for node in sorted_nodes if node[0] == 'h'] + output_nodes

    # Create dictionary mapping nodes to their index in sorted_nodes
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

    # Initialize list of node value tensors
    node_values = [None] * len(sorted_nodes)

    # Set the value of the input nodes
    for i in range(len(input_nodes)):
        node_values[i] = input[i].clone().detach()

    # Edges in order of when they should be processed
    sorted_edges = []
    for node in sorted_nodes:
        sorted_edges.extend(G.in_edges(node))

    # Get weight tensor for edges
    edge_weights = torch.tensor([G.get_edge_data(*edge)['weight'] for edge in sorted_edges], requires_grad=True)
    
    # Get the indices of the nodes connected by the edges
    sorted_edge_indices = list(map(lambda e: (node_to_index[e[0]], node_to_index[e[1]]), sorted_edges))

    # Create a dictionary mapping edges to their indices
    edge_to_index = {edge: idx for idx, edge in enumerate(sorted_edges)}

    # Forward pass
    for node in sorted_nodes[len(input_nodes):]:
        incoming_edges = list(G.in_edges(node))
        if incoming_edges:
            incoming_products = torch.stack([node_values[node_to_index[edge[0]]] * edge_weights[edge_to_index[edge]] for edge in incoming_edges])
            node_values[node_to_index[node]] = torch.relu(torch.sum(incoming_products)).requires_grad_(True)
        else:
            node_values[node_to_index[node]] = torch.tensor(0.0, requires_grad=True)
    for i, edge_indices in enumerate(sorted_edge_indices):
        node_values[edge_indices[1]] = node_values[edge_indices[0]] * edge_weights[i]

    # Stack output node values into a single tensor
    output_tensor = torch.stack([node_values[node_to_index[node]] for node in output_nodes])

    # Calculate cross entropy loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output_tensor.unsqueeze(0), torch.tensor([label]))

    # Backpropagate
    loss.backward()



    # Add 1/n to delta_utility for each output node
    for output_node in output_nodes:
        edges = G.in_edges(output_node)
        for edge in edges:
            G[edge[0]][edge[1]]['delta_utility'] = 1.0 / len(output_nodes)

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



    # Find node with largest hunger
    max_hunger_node = max(sorted_nodes[len(input_nodes):], key=lambda n: G.nodes[n]['hunger'])

    # Get all nodes that are not ancestors or descendants of max_hunger_node
    ancestors = nx.ancestors(G, max_hunger_node)
    descendants = nx.descendants(G, max_hunger_node)
    unrelated_nodes = set(G.nodes()) - ancestors - descendants - {max_hunger_node}

    



    print(weight_update)

print(trainset[0][0].reshape(-1)[0], trainset[0][1])
predict_sample(trainset[0][0].reshape(-1), trainset[0][1])