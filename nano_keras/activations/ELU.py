import numpy as np
from nano_keras.activations import Activation


class ELU(Activation):
    """ELU activation function
    """

    def __init__(self, alpha: float = 0.2):
        """Initalizer for ELU activation function

        Args:
            alpha (float, optional): By how much do we multiply np.exp(X) if X is smaller than 0. Defaults to 0.2.
        """
        super(ELU, self).__init__()
        self.alpha: float = alpha

    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply ELU activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.where(X > 0, X, self.alpha * np.exp(X) - 1)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of ELU activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.where(X >= 0, 1, self.alpha * np.exp(X))
