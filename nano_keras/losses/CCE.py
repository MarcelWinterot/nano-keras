import numpy as np
from nano_keras.losses import Loss


class CCE(Loss):
    def __init__(self) -> None:
        self.e: float = 1e-7

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Categorical cross entropy implementation of compute_loss function

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed loss
        """
        return -np.sum(yTrue * np.log(yPred + self.e))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Categorical cross entropy implementation of derivative of compute_loss, which computes the gradient used during backpropagation

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed gradient
        """
        yPred = np.sum(yPred, keepdims=True)
        yPred = np.clip(yPred, self.e, 1.0-self.e)
        return np.sum(yTrue * np.log(yPred))
