import numpy as np
from nano_keras.optimizers import Optimizer


class Adam(Optimizer):
    """Adam(Adaptive Moment Estimator) optimizer implementation of Optimizer class. You can read more about it at https://arxiv.org/pdf/1412.6980.pdf
    """    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7, adjust_biases_shape: bool = False) -> None:
        """Intializer to the Adam(Adaptive Moment Estimator) optimizer.

        Args:
            learning_rate (float, optional): Paramter that specifies how fast the model will learn. Defaults to 0.001.
            beta1 (float, optional): Paramter that controls the exponential moving average of the first moment of the gradient. Defaults to 0.9.
            beta2 (float, optional): Paramter that contorls the exponential moving average of the second moment of the gradient. Defaults to 0.999.
            epsilon (float, optional): Paramter that ensures we don't divide by 0 and adds numerical stability to learning rate. Defaults to 1e-7.
            adjust_biases_shape (bool, optional): Paramter that controles wheter we adjuts the bias gradients and moving averages for biases shapes. Default to False.
        """
        self.learning_rate: float = learning_rate
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.e: float = epsilon
        self.adjust_biases_shape: bool = adjust_biases_shape
        self.m_w: np.ndarray = np.array([])
        self.v_w: np.ndarray = np.array([])
        self.m_b: np.ndarray = np.array([])
        self.v_b: np.ndarray = np.array([])
        self.t: int = 0

    def apply_gradients(self, weightGradients: np.ndarray, biasGradients: np.ndarray, weights: np.ndarray, biases: np.ndarray, update_biases: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates params using provided gradients and Adam algorithm. You can read more about it
        at https://arxiv.org/pdf/1412.6980.pdf

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
        beta1T = self.beta1 ** self.t
        beta2T = self.beta2 ** self.t

        if self.m_w.size == 0:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)

        target_shape = weights.shape

        # Adjusting shapes before calculations
        slices = [slice(0, shape) for shape in target_shape]

        self.m_w = self._fill_array(self.m_w, target_shape)[tuple(slices)]
        self.v_w = self._fill_array(self.v_w, target_shape)[tuple(slices)]

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weightGradients
        self.v_w = self.beta2 * self.v_w + \
            (1 - self.beta2) * weightGradients ** 2

        m_hat_w = self.m_w / (1 - beta1T)
        v_hat_w = self.v_w / (1 - beta2T)

        weights += self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.e)

        if update_biases:
            if self.adjust_biases_shape:
                target_shape = biases.shape
                self.m_b = self._fill_array(self.m_b, target_shape)[
                    :target_shape[0]]
                self.v_b = self._fill_array(self.v_b, target_shape)[
                    :target_shape[0]]

            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * biasGradients
            self.v_b = self.beta2 * self.v_b + \
                (1 - self.beta2) * biasGradients ** 2

            m_hat_b = self.m_b / (1 - beta1T)
            v_hat_b = self.v_b / (1 - beta2T)

            biases += self.learning_rate * \
                m_hat_b / (np.sqrt(v_hat_b) + self.e)

        return (weights, biases)
