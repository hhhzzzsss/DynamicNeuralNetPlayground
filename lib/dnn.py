import torch
import numpy as np
from dnn_libs.agraph import AcyclicGraph

class DynamicNeuralNetwork:
    def __init__(self, input_size, output_size, utility_decay=0.01, hunger_decay=0.05, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.utility_decay = utility_decay
        self.hunger_decay = hunger_decay
        self.learning_rate = learning_rate
        
        self.G = AcyclicGraph()

        # Initialize nodes
        self.input_nodes = []
        self.output_nodes = []
        self.hidden_nodes = []
        for _ in range(input_size):
            node_id = self.G.add_node(hunger=0.0, utility=0.0)
            self.input_nodes.append(node_id)
        for _ in range(output_size):
            node_id = self.G.add_node(hunger=0.0, utility=0.0)
            self.output_nodes.append(node_id)

        self.input_nodes_set = set(self.input_nodes)
        self.output_nodes_set = set(self.output_nodes)
        self.hidden_nodes_set = set(self.hidden_nodes)

        self.edge_growth_intents = {}

    def add_hidden_node(self):
        node_id = self.G.add_node(hunger=0.0, utility=0.0)
        self.hidden_nodes.append(node_id)
        return node_id
    def add_edge(self, from_node, to_node, sign=0.0):
        self.G.add_edge(from_node, to_node, 0.01*sign)
    def add_double_edge(self, from_node, to_node, sign=0.0):
        hidden_node = self.add_hidden_node()
        self.G.add_edge(from_node, hidden_node, 1.0)
        self.add_edge(hidden_node, to_node, sign)

    # Expects input tensor of shape (batch_size, input_size)
    # Expects label tensor of shape (batch_size)
    def train_batch(self, input, label, grow_edge=True):
        # Assert input and label are PyTorch tensors
        assert isinstance(input, torch.Tensor), "Input must be a PyTorch tensor"
        assert isinstance(label, torch.Tensor), "Label must be a PyTorch tensor"

        # Check dimensions of input and label tensors
        assert input.shape[1] == self.input_size, f"Expected input.shape[1] to be {self.input_size}, but got {input.shape[1]}"
        assert len(input.shape) == 2, f"Expected input to have 2 dimensions, but got {len(input.shape)}"
        assert len(label.shape) == 1, f"Expected label to have 1 dimension, but got {len(label.shape)}"
        assert input.shape[0] == label.shape[0], f"Expected input.shape[0] and label.shape[0] to be equal, but got {input.shape[0]} and {label.shape[0]}"

        batch_size = input.shape[0]

        sorted_nodes = self.G.get_topological_order()
        sorted_edges = self.G.get_sorted_edges()

        # Initialize a list to hold the node values and edge weights as pytorch tensors
        pth_nodes = [None] * len(sorted_nodes)
        edge_weights = torch.tensor([weight for _, _, weight in sorted_edges], dtype=torch.float32, requires_grad=True)
        edge_index_map = {
            (node_from, node_to): i for i, (node_from, node_to, _) in enumerate(sorted_edges)
        }

        # Initialize the input nodes with the input tensor
        for node_id, input_slice in zip(self.input_nodes, input.transpose(0, 1)):
            pth_nodes[self.G.get_topological_index(node_id)] = input_slice
        
        # Forward pass
        for i in range(len(sorted_nodes)):
            if pth_nodes[i] is not None:
                continue
            node_id = sorted_nodes[i]
            in_neighbors = self.G.get_in_neighbors(node_id)
            if len(in_neighbors) > 0:
                incoming_products = torch.stack([
                    pth_nodes[self.G.get_topological_index(in_n)] * edge_weights[edge_index_map[(in_n, node_id)]]
                    for in_n in in_neighbors
                ])
                pth_nodes[i] = torch.sum(incoming_products, dim=0)
                if node_id not in self.output_nodes_set:
                    pth_nodes[i] = torch.relu(pth_nodes[i])
                pth_nodes[i].retain_grad()
            else:
                pth_nodes[i] = torch.zeros(batch_size, dtype=torch.float32, requires_grad=True)
                pth_nodes[i].retain_grad()

        output_tensor = torch.stack([
            pth_nodes[self.G.get_topological_index(node_id)] for node_id in self.output_nodes
        ], dim=1)
        input_tensor = torch.stack([
            pth_nodes[self.G.get_topological_index(node_id)] for node_id in self.input_nodes
        ], dim=1)
        # print(input_tensor)
        # print(f"output tensor:\n{output_tensor}")

        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output_tensor, label)

        # Backward pass
        loss.backward()

        ## Utility is propagated backwards by distributing the utility of the current node
        ## to all of its predecessors weighted by the edge-weight times value of the predecessor

        # Initialize delta_utility for all nodes
        delta_utility = {node_id: torch.zeros(batch_size, dtype=torch.float32) for node_id in sorted_nodes}

        # Add 1/n to delta_utility for each output node
        for node_id in self.output_nodes:
            delta_utility[node_id] = torch.ones(batch_size, dtype=torch.float32) / len(self.output_nodes)
        
        # Compute the delta utilities of each node in revesed topological order
        for node_id in reversed(sorted_nodes):
            # Update node utility
            current_utility = self.G.get_node_utility(node_id)
            current_utility += torch.mean(delta_utility[node_id])
            current_utility *= (1.0 - self.utility_decay)
            self.G.set_node_utility(node_id, current_utility)

            # Propagate utility backwards
            in_neighbors = self.G.get_in_neighbors(node_id)
            # Get total weighted input for normalization
            total_weight_value = sum(
                torch.abs(self.G.get_edge_weight(in_n, node_id) * pth_nodes[self.G.get_topological_index(in_n)])
                for in_n in in_neighbors
            )
            # Avoid division by zero
            total_weight_value += 1e-10
            # Distribute delta_utility
            for in_n in in_neighbors:
                weight = self.G.get_edge_weight(in_n, node_id)
                value = pth_nodes[self.G.get_topological_index(in_n)]
                proportion = abs(weight * value) / total_weight_value

                delta_utility[in_n] += delta_utility[node_id] * proportion
                
        ## Hunger is computed as the absolute value of the partial derivative of the loss
        ## with respect to the node's value

        # Update hunger
        for node_id in self.hidden_nodes + self.output_nodes:
            current_hunger = self.G.get_node_hunger(node_id)
            current_hunger += torch.mean(torch.abs(
                pth_nodes[self.G.get_topological_index(node_id)].grad
            ))
            current_hunger *= (1.0 - self.hunger_decay)
            self.G.set_node_hunger(node_id, current_hunger)

        # Update edge weights
        if len(sorted_edges) > 0:
            weight_update = -edge_weights.grad/torch.linalg.norm(edge_weights.grad) * self.learning_rate
            # for i, (_, to_node, _) in enumerate(sorted_edges):
            #     weight_update[i] /= (self.G.get_node_utility(to_node) + 1.0)
            self.G.set_edge_weights([
                (from_node, to_node, new_weight) for (from_node, to_node, _), new_weight
                in zip(sorted_edges, edge_weights + weight_update)
            ])

        # Get nodes with at most average in-degree
        non_input_nodes = self.hidden_nodes + self.output_nodes
        in_degrees = [self.G.get_in_degree(node_id) for node_id in non_input_nodes]
        avg_in_degree = np.mean(in_degrees)
        below_avg_nodes = [
            node_id for node_id in non_input_nodes if self.G.get_in_degree(node_id) <= avg_in_degree
        ]

        # Update edge growth intents
        max_hunger_node = max(below_avg_nodes, key=lambda node_id: self.G.get_node_hunger(node_id))
        if max_hunger_node:
            print(f"Max hunger node: {max_hunger_node} with hunger {self.G.get_node_hunger(max_hunger_node)}")
            in_neighbors = set(self.G.get_in_neighbors(max_hunger_node))
            out_neighbors = set(self.G.get_out_neighbors(max_hunger_node))
            successors = self.G.get_successors(max_hunger_node)
            all_nodes = set(self.G.get_nodes())
            potential_in_neighbors = all_nodes - in_neighbors - out_neighbors - successors - self.output_nodes_set - { max_hunger_node }
        
        for potential_in_neighbor in potential_in_neighbors:
            edge_id = (potential_in_neighbor, max_hunger_node)
            self.edge_growth_intents[edge_id] = \
                self.edge_growth_intents.get(edge_id, 0) \
                + torch.mean(pth_nodes[potential_in_neighbor] * pth_nodes[max_hunger_node].grad)
        
        if grow_edge and len(self.edge_growth_intents) > 0:
            self.grow_edge()

        return loss.item()
        
    def grow_edge(self):
        largest_intent_edge, intent_value = max(self.edge_growth_intents.items(), key=lambda x: abs(x[1]))
        from_node = largest_intent_edge[0]
        to_node = largest_intent_edge[1]
        sign = 1 if intent_value >= 0 else -1
        if self.G.get_node_utility(to_node) > 1:
            print(f"Growing double edge from {from_node} to {to_node} with intent {intent_value}")
            self.add_double_edge(from_node, to_node, sign)
        else:
            print(f"Growing single edge from {from_node} to {to_node} with intent {intent_value}")
            self.add_edge(from_node, to_node, sign)
        self.edge_growth_intents.clear()

    def draw(self, plt, ax):
        ax.clear()

        max_rank = 0
        ranks = {}

        sorted_nodes = self.G.get_topological_order()
        sorted_edges = self.G.get_sorted_edges()

        # Compute ranks for each node
        for node_id in self.input_nodes:
            ranks[node_id] = 0
        
        for node_id in sorted_nodes:
            if node_id in self.input_nodes_set:
                continue
            in_neighbors = self.G.get_in_neighbors(node_id)
            if len(in_neighbors) > 0:
                ranks[node_id] = max(ranks[in_n] for in_n in in_neighbors) + 1
                max_rank = max(max_rank, ranks[node_id])
            elif node_id not in self.output_nodes_set:
                raise RuntimeError("Non-output nodes should have at least one in-neighbor")
                
        for node_id in self.output_nodes:
            ranks[node_id] = max_rank + 1

        # Group nodes by rank
        rank_to_nodes = {}
        for node_id, rank in ranks.items():
            if rank not in rank_to_nodes:
                rank_to_nodes[rank] = []
            rank_to_nodes[rank].append(node_id)
        
        # Define positions for each node
        pos = {}
        for rank, nodes in rank_to_nodes.items():
            x = rank
            if rank == 0 or rank == max_rank+1:
                y = np.linspace(0, 1, len(nodes))
            else:
                y = np.linspace(0, 1, len(nodes)+2)[1:-1]
            for i, node_id in enumerate(nodes):
                pos[node_id] = (x, y[i])

        # Define colors for each node
        colors = {}
        for node_id in sorted_nodes:
            if node_id in self.input_nodes_set:
                colors[node_id] = 'blue'
            elif node_id in self.output_nodes_set:
                colors[node_id] = 'red'
            else:
                colors[node_id] = 'green'
        
        # Draw nodes
        for node_id, (x, y) in pos.items():
            ax.scatter(x, y, color=colors[node_id], s=100)
            # Add topological index as label for each node
            topo_index = self.G.get_topological_index(node_id)
            hunger = self.G.get_node_hunger(node_id)
            utility = self.G.get_node_utility(node_id)
            ax.text(x, y, f"{topo_index}\nid: {node_id}\nU: {utility:.2f}\nH: {hunger:.2f}", 
                ha='center', va='center', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        # Draw edges as arrows
        for from_node, to_node, weight in sorted_edges:
            x1, y1 = pos[from_node]
            x2, y2 = pos[to_node]
            ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.03, head_length=0.05, 
                fc='black', ec='black', length_includes_head=True)
            # Add weight label on each edge
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            # Format weight to 2 decimal places
            weight_label = f"{weight:.2f}"
            # Create a small white background for better readability
            ax.text(midpoint_x, midpoint_y, weight_label, 
                ha='center', va='center', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))

        plt.draw()
        plt.pause(0.001)
