"""
Zero initialization

There ate two types of parameters to initialize in a neural network:

    - the weight matrices (W^[1], W^[2], W^[3], ..., W^[L-1], W^[L]])
    - the bias vectors (b^[1], b^[2], b^[3], ..., b^[L-1], b^[L]])

Implement the following function to initialize all parameters to zeros.
You'll see later that this does not work well since it fails to "break symmetry",
but lets try it anyway and see what happens. Use np.zeros((..,..)) with the correct shapes.
"""
import numpy as np


def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
    b1 -- bias vector of shape (layers_dims[1], 1)
    ...
    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
    bL -- bias vector of shape (layers_dims[L], 1)
    """

    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


"""
The model is predicting 0 for every example.

In general, initializing all the weights to zero results in the network failing to break symmetry.
This means that every neuron in each layer will learn the same thing, and you might as well be training
a neural network with  n^[l]=1  for every layer, and the network is no more powerful than a linear classifier
 such as logistic regression.
 
What you should remember:

    - The weights  W^[l]  should be initialized randomly to break symmetry.
    - It is however okay to initialize the biases  b[l]b[l]  to zeros. 
      Symmetry is still broken so long as  W[l]W[l]  is initialized randomly.
"""