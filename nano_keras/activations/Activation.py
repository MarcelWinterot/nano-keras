import numpy as np


class Activation:
    def __init__(self) -> None:
        """Initalizer for most activation function that don't have any special params like LeakyReLU or ELU
        """
        pass

    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Base activation class implementation of compute loss function

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        pass

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Base activation class implementation of the derivative of apply_activation

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them        
        """
        pass
