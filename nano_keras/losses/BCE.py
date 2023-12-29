import numpy as np
from nano_keras.losses import Loss


class BCE(Loss):
    """Binary cross entropy implementation of Loss class
    """

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Binary cross entropy implementation of compute_loss function

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed loss
        """
        yPred = np.clip(yPred, self.e, 1 - self.e)
        result = (1 - yTrue) * np.log(1 - yPred + self.e) + \
            yTrue * np.log(yPred + self.e)
        return -np.mean(result)

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        """Binary cross entropy implementation of derivative of compute_loss, which computes the gradient used during backpropagation

        Args:
            yTrue (np.ndarray): y dataset to which we compare model predictions
            yPred (np.ndarray): model predictions on x dataset corresponding to y

        Returns:
            np.ndarray: computed gradient
        """
        yPred = np.clip(yPred, self.e, 1 - self.e)
        result = yTrue * np.log(yPred + self.e)
        result += (1 - yTrue) * np.log(1 - yPred + self.e)
        return -result
