import numpy as np
from nano_keras.activations import Activation


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.2):
        """Initalizer for LeakyRELU activation function

        Args:
            alpha (float, optional): By how much do we multiply X if it's is smaller than 0. Defaults to 0.2.
        """
        self.alpha: float = alpha

    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply LeakyReLU activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.maximum(self.alpha*X, X)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of LeakyReLU activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.where(X <= 0, 1, self.alpha)
