import numpy as np
from nano_keras.regulizers import Regularizer


class L1(Regularizer):
    """L1 Regularizer implementation of Regularizer.
    """

    def update_gradient(self, gradient: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """L1 regulizer implementation of the gradient update functiion

        Args:
            gradient (float): gradient calculate by loss.compute_derivative or previous layers output gradient calculation
            weights (np.ndarray): models weights
            biases (np.ndarray): models biases

        Returns:
            np.ndarray: updated gradient
        """
        gradient += self.strength * \
            np.sum(np.abs(weights)) + self.strength * np.sum(np.abs(biases))
        return gradient
