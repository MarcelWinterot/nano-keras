import numpy as np
from nano_keras.activations import Activation


class Softmax(Activation):
    """Softmax activation function
    """

    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply softmax activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        ex = np.exp(X - np.max(X) + self.e)
        return ex / ex.sum(axis=0)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of softmax activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        s = self.apply_activation(X)
        return np.diag(s) - np.outer(s, s)
