import numpy as np
from nano_keras.losses import Loss


class Huber(Loss):
    def __init__(self, delta: float = 0.5) -> None:
        """Initalizer for huber loss class

        Args:
            delta (float, optional): Determines the transition point between the quadratic and linear regions of the loss function. Defaults to 0.5.
        """
        self.delta: float = delta

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Huber implementation of compute_loss function

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed loss
        """
        absolute_error = np.abs(yTrue - yPred)
        return np.mean(np.where(absolute_error <= self.delta, 0.5 * (absolute_error**2), self.delta * (absolute_error - 0.5 * self.delta)))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Huber implementation of derivative of compute_loss, which computes the gradient used during backpropagation

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed gradient
        """
        absolute_error = np.abs(yTrue - yPred)
        return np.where(absolute_error <= self.delta,
                        yPred - yTrue, self.delta * np.sign(yPred - yTrue))
