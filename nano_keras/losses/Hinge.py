import numpy as np
from nano_keras.losses import Loss


class Hinge(Loss):
    """Hinge implementation of Loss class
    """

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Hinge implementation of compute_loss function

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed loss
        """
        return np.mean(np.maximum(0, 1 - yTrue * yPred))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Hinge implementation of derivative of compute_loss, which computes the gradient used during backpropagation

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed gradient
        """
        return -yTrue * (1 - yTrue * yPred > 0).astype(float)
