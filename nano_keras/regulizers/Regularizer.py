import numpy as np


class Regularizer:
    """Regularizer base class.
    """

    def __init__(self, strength: float = 1e-3):
        """Intializer for the base Regulzer implementation

        Args:
            strength (float, optional): How much do we punish the model for having big weights and biases. Defaults to 1e-3.
        """
        self.strength: float = strength

    def update_gradient(self, gradient: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """Base implementation for gradient update function of Regulizer class

        Args:
            gradient (float): gradient calculate by loss.compute_derivative or previous layers output gradient calculation
            weights (np.ndarray): models weights
            biases (np.ndarray): models biases

        Returns:
            float: updated gradient
        """
        pass
