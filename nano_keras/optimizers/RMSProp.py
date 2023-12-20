import numpy as np
from nano_keras.optimizers import Optimizer


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-7, adjust_biases_shape: bool = False) -> None:
        """Initalizer for the RMSProp(Root Mean Square Propagation) algorithm.

        Args:
            learning_rate (float, optional): Paramter that specifies how fast the model will learn. Defaults to 0.001.
            rho (float, optional): Paramter that controls the exponential moving average of the squared gradients. Defaults to 0.9.
            epsilon (float, optional): Paramter that ensures we don't divide by 0 and adds numerical stability to learning rate. Defaults to 1e-7.
            adjust_biases_shape (bool, optional): Paramter that controles wheter we adjuts the bias gradients and moving averages for biases shapes. Default to False.
        """
        self.learning_rate: float = learning_rate
        self.rho: float = rho
        self.e: float = epsilon
        self.adjust_biases_shape: bool = adjust_biases_shape
        self.v_w: np.ndarray = np.array([])
        self.v_b: np.ndarray = np.array([])

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray, update_biases: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates models weights and biases using the RMSprop algorithm. 

        Args:
            weights_gradients (np.ndarray): Weight gradients you've calculated
            bias_gradients (np.ndarray): Bias gradients you've calculated
            weights (np.ndarray): Model or layers weights you want to update
            biases (np.ndarray): Model or layers biases you want to update
            update_biases (bool): Parameter that controls whether the biases should be updated. Defaults to True

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated weights and biases. First element are the weights and second are the biases.
        """
        if self.v_w.size == 0:
            self.v_w = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)

        # Adjusting shapes before calculations
        target_shape = weights.shape

        slices = [slice(0, shape) for shape in target_shape]

        self.v_w = self._fill_array(self.v_w, target_shape)[tuple(slices)]

        # Calculations
        self.v_w = self.rho * self.v_w + (1 - self.rho) * weights_gradients**2

        weights += self.learning_rate * \
            (weights_gradients/(np.sqrt(self.v_w) + self.e))

        if update_biases:
            if self.adjust_biases_shape:
                target_shape = biases.shape
                self.v_b = self._fill_array(self.v_b, target_shape)[
                    :target_shape[0]]

            self.v_b = self.rho * self.v_b + (1 - self.rho) * bias_gradients**2

            biases += self.learning_rate * \
                (bias_gradients / (np.sqrt(self.v_b) + self.e))

        return (weights, biases)
