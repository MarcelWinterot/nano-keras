import numpy as np
from nano_keras.optimizers import Optimizer


class Ftrl(Optimizer):
    """Ftrl(Follow The Regularized Leader) implementation of Optimizer class. You can read more about it at https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf 
    """

    def __init__(self, learning_rate: float = 0.001, learning_rate_power: float = 0.5, beta: float = 0.0, l1_regularization_strength: float = 0.0, adjust_biases_shape: bool = False) -> None:
        """Intializer to the Ftrl(Follow The Regularized Leader) optimizer.

        Args:
            learning_rate (float, optional): Paramter that specifies how fast the model will learn. Defaults to 0.001.
            learning_rate_power (float, optional): Controls how the learning rate decreases during training.. Defaults to 0.5.
            beta (float, optional): Beta value representing the beta value from the paper. Defaults to 0.0.
            l1_regularization_strength (float, optional): Strength of the l1 regularization. Defaults to 0.0.
            adjust_biases_shape (bool, optional): Paramter that controles wheter we adjuts the bias gradients and moving averages for biases shapes. Default to False.
        """
        self.learning_rate = learning_rate
        self.learning_rate_power = learning_rate_power
        self.beta = beta
        self.l1_regularization_strength = l1_regularization_strength
        self.adjust_biases_shape = adjust_biases_shape
        self.e = 1e-7

        self.z = None
        self.n = None

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray, update_biases: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if self.z is None:
            self.z = np.zeros_like(weights)
            self.n = np.zeros_like(weights)

        # Adjusting the shapes
        target_shape = weights.shape

        slices = [slice(0, shape) for shape in target_shape]

        self.z = self._fill_array(self.z, target_shape)[tuple(slices)]
        self.n = self._fill_array(self.n, target_shape)[tuple(slices)]

        # Calculations

        self.z += weights_gradients - self.learning_rate * weights
        self.n += weights_gradients ** 2

        g = -1 * (self.z / (self.n ** self.learning_rate_power + self.beta + self.e))
        weights += np.where(np.abs(self.z) <= self.l1_regularization_strength,
                            weights, g / self.learning_rate)

        if update_biases:
            biases += bias_gradients

        return weights, biases
