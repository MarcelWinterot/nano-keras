import numpy as np
from nano_keras.regulizers import Regularizer


class L1L2(Regularizer):
    """L1L2 Regularizer implementation of Regularizer.
    """
    def __init__(self, l1_strength: float = 1e-3, l2_strength: float = 1e-3):
        """Intailizer for the L1L2 regulizer

        Args:
            l1_strength (float, optional): How much do we punish the model for having big weights and biases using L1. Defaults to 1e-3.
            l2_strength (float, optional): How much do we punish the model for having big weights and biases using L2. Defaults to 1e-3.
        """
        self.l1_strength: float = l1_strength
        self.l2_strength: float = l2_strength

    def update_gradient(self, gradient: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """L1L2 regulizer implementation of the gradient update function

        Args:
            gradient (float): gradient calculate by loss.compute_derivative or previous layers output gradient calculation
            weights (np.ndarray): models weights
            biases (np.ndarray): models biases

        Returns:
            float: updated gradient
        """
        gradient += self.l1_strength * np.sum(np.abs(weights)) + self.l1_strength * np.sum(np.abs(
            biases)) + self.l2_strength * np.sum(np.square(weights)) + self.l2_strength * np.sum(np.square(biases))
        return gradient
