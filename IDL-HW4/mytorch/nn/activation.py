import numpy as np
from itertools import product


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        self.Z = Z
        Z_stable = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp = np.exp(Z_stable)
        A = exp / np.sum(exp, axis=self.dim, keepdims=True)
        self.A = A
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """

        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]

        loss_in = dLdA
        softmax_out = self.A

        # Reshape input to 2D
        if len(shape) > 2:
            loss_in = np.moveaxis(dLdA, self.dim, -1)
            loss_in = loss_in.reshape((-1, C))
            softmax_out = self.A.reshape((-1, C))

        # How each element in specified dimension affects others
        jacobian = np.zeros((C, C))
        dLdZ = np.zeros_like(loss_in)
        indices = list(product(np.arange(softmax_out.shape[1]), np.arange(softmax_out.shape[1])))

        # Iterate over the other dims
        for dim_idx in range(softmax_out.shape[0]):
            # Get a C dim vector
            curr_vec = softmax_out[dim_idx, :]
            # for each combination of members in the c dim vector to fill the C, C Jacobian
            for m, n in indices:
                if m == n:
                    jacobian[m, n] = curr_vec[m] * (1 - curr_vec[m])
                else:
                    jacobian[m, n] = -1 * curr_vec[m] * curr_vec[n]

            # X, C = (X, C) (C, C)
            dLdZ[dim_idx] = loss_in[dim_idx, :] @ jacobian

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            dLdZ = dLdZ.reshape((*shape[:-1], C))
            dLdZ = np.moveaxis(dLdZ, -1, self.dim)

        return dLdZ
 

