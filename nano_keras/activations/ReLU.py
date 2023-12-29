import numpy as np
from nano_keras.activations import Activation


class ReLU(Activation):
    """ReLU activation function
    """    
    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply ReLU activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        super(ReLU, self).__init__()
        return np.maximum(0.0, X)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of tanh activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.where(X < 0, 0, 1)
