import numpy as np
from nano_keras.activations import Activation


class Tanh(Activation):
    """Tanh activation function
    """

    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply tanh activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        output = (np.exp(X) - np.exp(-X)) / \
            np.nan_to_num(np.exp(X) + np.exp(-X) + self.e, copy=False)
        return output

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of tanh activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return 1 - self.apply_activation(X) ** 2
