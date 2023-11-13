import numpy as np


class Regularizer:
    def __init__(self, strength: float = 1e-3):
        """Intializer for the base Regulzer implementation

        Args:
            strength (float, optional): How much do we punish the model for having big weights and biases. Defaults to 1e-3.
        """
        self.strength = strength

    def compute_loss(self, gradient: float, weights: np.ndarray, biases: np.ndarray) -> float:
        """Base implementation for compute_loss function of Regulizer

        Args:
            gradient (float): gradient of the model
            weights (np.ndarray): models weights
            biases (np.ndarray): models biases

        Returns:
            float: new gradient
        """
        pass


class L1(Regularizer):
    def compute_loss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        """L1 regulizer implementation for compute_loss function of Regulizer

        Args:
            gradient (float): gradient of the model
            weights (np.ndarray): models weights
            biases (np.ndarray): models biases

        Returns:
            float: new gradient
        """
        loss += self.strength * \
            np.sum(np.abs(weights)) + self.strength * np.sum(np.abs(biases))
        return loss


class L2(Regularizer):
    def compute_loss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        """L2 regulizer implementation for compute_loss function of Regulizer

        Args:
            gradient (float): gradient of the model
            weights (np.ndarray): models weights
            biases (np.ndarray): models biases

        Returns:
            float: new gradient
        """
        loss += self.strength * \
            np.sum(np.square(weights)) + self.strength * \
            np.sum(np.square(biases))
        return loss


class L1L2(Regularizer):
    def __init__(self, l1_strength: float = 1e-3, l2_strength: float = 1e-3):
        """Intailizer for the L1L2 regulizer

        Args:
            l1_strength (float, optional): How much do we punish the model for having big weights and biases using L1. Defaults to 1e-3.
            l2_strength (float, optional): How much do we punish the model for having big weights and biases using L2. Defaults to 1e-3.
        """
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

    def compute_loss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        """L1L2 regulizer implementation for compute_loss function of Regulizer

        Args:
            gradient (float): gradient of the model
            weights (np.ndarray): models weights
            biases (np.ndarray): models biases

        Returns:
            float: new gradient
        """
        loss += self.l1_strength * np.sum(np.abs(weights)) + self.l1_strength * np.sum(np.abs(
            biases)) + self.l2_strength * np.sum(np.square(weights)) + self.l2_strength * np.sum(np.square(biases))
        return loss
