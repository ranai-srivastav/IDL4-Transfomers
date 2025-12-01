import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        C, c = self.W.shape

        self.A = A
        self.input_batch_shape = A.shape[:-1]
        A = A.reshape((-1, c))
        Z = A @ self.W.T + self.b
        Z = Z.reshape((*self.input_batch_shape, C))
        return Z



    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """

        # dLdZ is of shape (*, C), * can be any number of dims
        # Reshape * so it is a 2D matrix of shape (X, C) where X is multiplication of *
        # Backprop through that, to get dZdW, dZdA, dZdB

        C, c = self.W.shape
        layer_in = self.A.reshape((-1, c))
        loss_in = dLdZ.reshape((-1, C))

        # Compute gradients (refer to the equations in the writeup)
                 #  (C, X) (X, c)
        self.dLdW = loss_in.T @ layer_in
        self.dLdb = np.sum(loss_in.T, axis=1)
                 #  (C, X) (C, c)
        self.dLdA = loss_in @ self.W
        self.dLdA = self.dLdA.reshape((*self.input_batch_shape, c))

        # Return gradient of loss wrt input
        return self.dLdA
