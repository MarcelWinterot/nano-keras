import numpy as np
from nano_keras.activations import Activation


class Sigmoid(Activation):
    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply sigmoid activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return 1 / (1 + np.exp(-X + 1e-7))

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of sigmoid activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        sX = self.apply_activation(X)
        return sX * (1 - sX)
