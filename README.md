# Dynamic Neural Network Playground
(In progress) experimenting with an unusual neural network architecture

Run `pip install .` to setup.

Then run `python train.py` to run.

## Basic Explanation
This is an experimental neural network architecture based on Rich Sutton's talk, [Toward a better Deep Learning](https://www.youtube.com/watch?v=YLiXgPhb8cQ).

It dynamically grows edges leading into the nodes with large hunger values. If the node has a high utility value, it grows a "double edge" instead of a regular edge. A double edge is just two edges with a hidden node between them. This is because a node with high utility is likely part of the "backbone", and the backbone should not be directly modified with additional edges, so instead, it creates extra nodes that lead into it.

Every hidden node has a ReLU activation, while the output nodes have a softmax activation.

The example training script currently trains a neural network to perform an xor operation between too input. The output is represented with two nodes because I implemented the model as a classification model. To compensate for the lack of a bias value for the neurons, I added an input node that always has a value of 1.

I also normalize the gradient because I don't have proper weight normalization for this model, which often leads to the gradients collapsing.

### Utility and Hunger
Hunger is based on the absolute value of the derivative of the loss with respect to each loss. The hunger of a node is simply calculated as an exponential moving average of the absolute value of the gradient.

Utility, on the other hand is calculated as follows: After the forward pass, we assign one divided by the number of output nodes as the delta_utility for each output node. The delta_utility values for all remaining nodes are initialized to 0. Then, for each node in reverse topological order, we, add a fraction of the delta_utility of the current node to all in-neighbors, weighted by the amount of value that node contributed to the current node's value in the forward pass. The sum of delta_utility added to each of the in-neighbors adds up to the delta_utility of the current node.

The utility of each node is calculated as an exponential moving average of the final delta_utility values of each node.
