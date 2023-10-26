import numpy as np
from typing import Union


class EarlyStopping:
    """
    For now it will be quite primitive watching over only the loss of the model, as that's
    the only variable I've currently implemented. In the future it will have most of the keras
    implementation functions
    """

    def __init__(self, patience: int, min_delta: float = 0.0001, restore_best_weights: bool = False) -> None:
        """Early stopping implementation using just python and numpy

        Args:
            patience (int): For how many epochs can the metrics degrade before stopping training
            min_delta (float): By how much must the metrics improve in order for it to be classiffied as an improvement
            restore_best_weights (bool, optional): Should the best weights be restored when the training is finished. Defaults to False.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.loss = np.inf
        self.weights = np.array([])
        self.biases = np.array([])
        self.counter = 0

    def monitor(self, loss: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> Union[tuple, None]:
        """
        Plans for the algorithm:
        We will check if the new loss has improved to the currently stored one
        If it has improved it will be stored in the memory with the current weights, and counter will be reset to 0
        If it has not improved we will update the counter by 1 and check if the counter == patience, if so end the training and return the best weights

        Args:
            loss (np.ndarray): New loss
            weights (np.ndarray): Our models weights
            biases (np.ndarray): Our models biases

        Returns:
            Union[tuple, None]: Either the weights and biases in a tuple if the training has finished or nothing if the training continues
        """
        if loss - self.loss < self.min_delta:
            self.loss = loss
            self.weights = weights
            self.biases = biases
            return None

        self.counter += 1
        if self.counter >= self.patience:
            print(
                f"Finshied the training process. {'Returning the weights and biases' if self.restore_best_weights else ''}")
            return (self.weights, self.biases) if self.restore_best_weights else ()
