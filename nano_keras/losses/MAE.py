import numpy as np
from nano_keras.losses import Loss


class MAE(Loss):
    """Mean absolute error implementation of Loss class
    """

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Mean absolute error implementation of compute_loss function

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed loss
        """
        return np.mean(abs(yTrue - yPred))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Mean absolut error implementation of derivative of compute_loss, which computes the gradient used during backpropagation

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed gradient
        """
        return -1 if yTrue - yPred >= 0 else 1
