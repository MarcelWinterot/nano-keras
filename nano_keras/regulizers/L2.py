import numpy as np
from nano_keras.regulizers import Regularizer


class L2(Regularizer):
    """L2 Regularizer implementation of Regularizer.
    """

    def update_gradient(self, gradient: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """L2 regulizer implementation of the gradient update function

        Args:
            gradient (float): gradient calculate by loss.compute_derivative or previous layers output gradient calculation
            weights (np.ndarray): models weights
            biases (np.ndarray): models biases

        Returns:
            float: updated gradient
        """
        gradient += self.strength * \
            np.sum(np.square(weights)) + self.strength * \
            np.sum(np.square(biases))
        return gradient
