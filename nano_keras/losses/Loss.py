import numpy as np


class Loss:
    """Base class for all Loss classes.
    """

    def __init__(self) -> None:
        """Initalizer for all Loss classes that don't have any paramters to set
        """
        self.e = 1e-7

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Base implementation of compute_loss function

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed loss
        """
        pass

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Base implementation of derivative of compute_loss, which computes the gradient used during backpropagation

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed gradient
        """
        pass
