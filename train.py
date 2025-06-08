import torch
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from lib.dnn import DynamicNeuralNetwork

def train_xor():
    # Create a DynamicNeuralNetwork instance
    dnn = DynamicNeuralNetwork(input_size=3, output_size=2)

    # Define the XOR dataset
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int64)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Initialize the plot
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    # Train the model
    for batch in range(1000):
        loss = dnn.train_batch(X_tensor, y_tensor, grow_edge=batch<20)
        print(f"Batch {batch}: Loss = {loss}")
        dnn.draw(plt, ax)
        # input()

train_xor()

print("Press Enter to exit...")
input()
