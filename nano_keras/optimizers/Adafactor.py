import numpy as np
import math
from nano_keras.optimizers import Optimizer


class Adafactor(Optimizer):
    def __init__(self, e_1: float = 10**-30, e_2: float = 0.001, d: float = 1, p: float = 0.01, beta_2: float = 0.999, adjust_biases_shape: bool = False) -> None:
        """Intializer to the Adam(Adaptive Moment Estimator) optimizer.

        Args:
            learning_rate (float, optional): Paramter that specifies how fast the model will learn. Defaults to 0.001.
            beta1 (float, optional): Paramter that controls the exponential moving average of the first moment of the gradient. Defaults to 0.9.
            beta2 (float, optional): Paramter that contorls the exponential moving average of the second moment of the gradient. Defaults to 0.999.
            epsilon (float, optional): Paramter that ensures we don't divide by 0 and adds numerical stability to learning rate. Defaults to 1e-7.
            adjust_biases_shape (bool, optional): Paramter that controles wheter we adjuts the bias gradients and moving averages for biases shapes. Default to False.
        """
        self.e_1 = e_1
        self.e_2 = e_2
        self.d = d
        self.p = p
        self.beta_2 = beta_2

        self.learning_rate = 0

        self.adjust_biases_shape: bool = adjust_biases_shape
        self.R_w: np.ndarray = np.array([])
        self.C_w: np.ndarray = np.array([])

        self.V_b: np.ndarray = np.array([])

        self.t: int = 0

    def RMS(self, x: np.ndarray, gradient: np.ndarray) -> float:
        output = np.mean(((gradient**2) / (x + self.e_1)))

        return np.sqrt(np.abs(output))

    def apply_gradients(self, weightGradients: np.ndarray, biasGradients: np.ndarray, weights: np.ndarray, biases: np.ndarray, update_biases: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates params using provided gradients and Adafactor algorithm. You can read more about it
        at https://arxiv.org/pdf/1804.04235.pdf

        Args:
            weights_gradients (np.ndarray): Weight gradients you've calculated
            bias_gradients (np.ndarray): Bias gradients you've calculated
            weights (np.ndarray): Model or layers weights you want to update
            biases (np.ndarray): Model or layers biases you want to update
            update_biases (bool): Parameter that controls whether the biases should be updated. Defaults to True

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated weights and biases. First element are the weights and second are the biases.
        """
        self.t += 1

        if self.R_w.size == 0:
            self.R_w = np.zeros_like(weights)
            self.C_w = np.zeros_like(weights)
            self.V_b = np.zeros_like(biases)

        target_shape = weights.shape

        slices = [slice(0, shape) for shape in target_shape]

        self.R_w = self._fill_array(self.R_w, target_shape)[tuple(slices)]
        self.C_w = self._fill_array(self.C_w, target_shape)[tuple(slices)]

        self.learning_rate = max(self.e_2, self.RMS(
            self.R_w, weightGradients)) * self.p

        self.R_w = self.beta_2 * self.R_w + \
            (1 - self.beta_2) * (weightGradients**2)

        self.C_w = self.beta_2 * self.C_w + \
            (1 - self.beta_2) * (weightGradients ** 2)

        V_w = self.R_w * self.C_w / (self.R_w + self.e_1)

        U_w = weightGradients / np.sqrt(V_w + self.e_1)
        U_w = U_w / np.maximum(1.0, self.RMS(U_w, weightGradients) / self.d)

        weights += self.learning_rate * U_w

        if update_biases:
            if self.adjust_biases_shape:
                target_shape = biases.shape
                self.V_b = self._fill_array(self.V_b, target_shape)[
                    :target_shape[0]]

            self.learning_rate = max(self.e_2, self.RMS(
                biases, biasGradients)) * self.p

            self.V_b = self.beta_2 * self.V_b + \
                (1 - self.beta_2) * (biasGradients ** 2)

            U_b = biasGradients / np.sqrt(self.V_b + self.e_1)

            U_b = U_b / np.maximum(1.0, self.RMS(U_b, biasGradients) / self.d)

            biases += self.learning_rate * U_b

        return weights, biases
