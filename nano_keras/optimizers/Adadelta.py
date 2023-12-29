import numpy as np
from nano_keras.optimizers import Optimizer


class Adadelta(Optimizer):
    """Adadelta (Adaptive delta) optimizer implementation of Optimizer class. You can read more about it at https://arxiv.org/pdf/1212.5701.pdf
    """

    def __init__(self, rho: float = 0.9, epsilon: float = 1e-7, adjust_biases_shape: bool = False) -> None:
        """Initalizer for the Adadelta(Adaptive delta) algorithm.

        Args:
            rho (float, optional): Parameter that controls an exponential moving average. Defaults to 0.9.
            epsilon (float, optional): Paramter that ensures we don't divide by 0 and adds numerical stability to learning rate. Defaults to 1e-7.
            adjust_biases_shape (bool, optional): Paramter that controles wheter we adjuts the bias gradients and moving averages for biases shapes. Default to False.
        """
        self.rho: float = rho
        self.e: float = epsilon
        self.adjust_biases_shape: bool = adjust_biases_shape
        self.v_w: np.ndarray = np.array([])
        self.v_b: np.ndarray = np.array([])
        self.v_w_a: np.ndarray = np.array([])
        self.v_b_a: np.ndarray = np.array([])

        # Doing this as we don't want an error when calling callbacks
        self.learning_rate = 0

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray, update_biases: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates models weights and biases using the Adadelta algorithm. 

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
            self.v_w_a = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)
            self.v_b_a = np.zeros_like(biases)

        target_shape = weights.shape

        slices = [slice(0, shape) for shape in target_shape]

        self.v_w = self._fill_array(self.v_w, target_shape)[tuple(slices)]
        self.v_w_a = self._fill_array(self.v_w_a, target_shape)[tuple(slices)]

        self.v_w = self.rho * self.v_w + \
            (1 - self.rho) * weights_gradients ** 2

        # It should be -np.sqrt() but I've found that it works better without the minus probably because I'm calculating the gradient incorrectly
        weights_update = np.sqrt(self.v_w_a + self.e) / \
            np.sqrt(self.v_w + self.e) * weights_gradients

        weights += weights_update

        self.v_w_a = self.rho * self.v_w_a + \
            (1 - self.rho) * weights_update ** 2

        if update_biases:
            if self.adjust_biases_shape:
                target_shape = biases.shape
                self.v_b = self._fill_array(self.v_b, target_shape)[
                    :target_shape[0]]
                self.v_b_a = self._fill_array(self.v_b_a, target_shape)[
                    :target_shape[0]]

            self.v_b = self.rho * self.v_b + (1 - self.rho) + bias_gradients**2

            bias_update = np.sqrt(self.v_b_a + self.e) / \
                np.sqrt(self.v_b + self.e) * bias_gradients
            biases += bias_update

            self.v_b_a = self.rho * self.v_b_a + \
                (1 - self.rho) * bias_update ** 2

        return (weights, biases)
