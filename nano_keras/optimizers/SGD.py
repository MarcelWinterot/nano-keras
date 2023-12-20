import numpy as np
from nano_keras.optimizers import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        """Initializer for the SGD(Stochastic Gradient Descent) optimizer

        Args:
            learning_rate (float, optional): Parameter that spiecfies how fast the model should learn. Defaults to 0.001.
        """
        self.learning_rate: float = learning_rate

    def apply_gradients(self, weight_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray, update_biases: bool = True) -> tuple:
        """Function that updates params using provided gradients and SGD algorithm.

        Args:
            weightGradients (np.ndarray): calculated weights gradients
            biasGradients (np.ndarray): calculated bias gradients
            weights (np.ndarray): weights of the layer
            biases (np.ndarray): biases of the layer
            update_biases (bool): Parameter that controls whether the biases should be updated. Defaults to True

        Returns:
            tuple: a tuple containing new weights and biases
        """
        weights += self.learning_rate * weight_gradients
        if update_biases:
            biases += self.learning_rate * bias_gradients
        return (weights, biases)
