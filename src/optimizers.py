import numpy as np


class Optimizer:
    def __init__(self) -> None:
        pass

    def apply_gradients(self) -> None:
        pass


class Adam(Optimizer):
    def __init__(self, learningRate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7) -> None:
        """Intializer to the adaptive momentum estimator optimizer.

        Args:
            learningRate (float, optional): Paramter that specifies how fast the model will learn. Defaults to 0.001.
            beta1 (float, optional): Paramter that controls the exponential moving average of the first moment of the gradient. Defaults to 0.9.
            beta2 (float, optional): Paramter that contorls the exponential moving average of the second moment of the gradient. Defaults to 0.999.
            epsilon (float, optional): Paramter that ensures we don't divide by 0 and adds numerical stability to learning rate. Defaults to 1e-7.
        """
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = np.array([])
        self.v_w = np.array([])
        self.m_b = np.array([])
        self.v_b = np.array([])
        self.t = 0

    def fill_array_(self, arr: np.ndarray, targetShape: tuple, isBiases: bool = False) -> np.ndarray:
        """Support function to fill the m_w, v_w and m_b, v_b arrays if their shapes are smaller than the gradients

        Args:
            arr (np.ndarray): array to fill
            targetShape (tuple): what shape should the array have after filling it
            isBiases (bool, optional): If we expand the m_b and v_b we set it to True as bias gradients are 1D. Defaults to False.

        Returns:
            np.ndarray: filled array
        """
        arr_shape = arr.shape
        if not isBiases:
            paddingNeeded = (
                max(0, targetShape[0] - arr_shape[0]), max(0, targetShape[1] - arr_shape[1]))
            result = np.pad(
                arr, ((0, paddingNeeded[0]), (0, paddingNeeded[1])), mode='constant')
        else:
            paddingNeeded = max(0, targetShape[0] - arr_shape[0])
            result = np.pad(arr, ((0, paddingNeeded)), mode='constant')

        return result

    def apply_gradients(self, weightGradients: np.ndarray, biasGradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple:
        """Function that updates params using provided gradients and Adam algorithm. You can read more about it
        at https://arxiv.org/pdf/1412.6980.pdf

        Args:
            weightGradients (np.ndarray): calculated weights gradients
            biasGradients (np.ndarray): calculated bias gradients
            weights (np.ndarray): weights of the layer
            biases (np.ndarray): biases of the layer

        Returns:
            tuple: a tuple containing new weights and biases
        """
        self.t += 1
        beta1T = self.beta1 ** self.t
        beta2T = self.beta2 ** self.t

        if self.m_w.size == 0:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)

        targetShape = weights.shape
        self.m_w = self.beta1 * \
            self.fill_array_(self.m_w, targetShape)[:weightGradients.size] \
            + (1 - self.beta1) * weightGradients
        self.v_w = self.beta2 * \
            self.fill_array_(self.v_w, targetShape)[:weightGradients.size] + \
            (1 - self.beta2) * weightGradients ** 2

        m_hat_w = self.m_w / (1 - beta1T)
        v_hat_w = self.v_w / (1 - beta2T)

        weights += self.learningRate * \
            m_hat_w[:targetShape[0], :targetShape[1]] / \
            (np.sqrt(v_hat_w[:targetShape[0],
             :targetShape[1]]) + self.epsilon)

        targetShape = biases.shape
        self.m_b = self.beta1 * \
            self.fill_array_(self.m_b, targetShape, True)[
                :biasGradients.size] + (1 - self.beta1) * biasGradients
        self.v_b = self.beta2 * \
            self.fill_array_(self.v_b, targetShape, True)[:biasGradients.size] + \
            (1 - self.beta2) * biasGradients ** 2

        m_hat_b = self.m_b / (1 - beta1T)
        v_hat_b = self.v_b / (1 - beta2T)

        biases += self.learningRate * m_hat_b / \
            (np.sqrt(v_hat_b) + self.epsilon)

        return (weights, biases)


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        """Initializer for the stochastic gradient descent optimizer

        Args:
            learning_rate (float, optional): Parameter that spiecfies how fast the model should learn. Defaults to 0.001.
        """
        self.learning_rate = learning_rate

    def apply_gradients(self, weight_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple:
        """Function that updates params using provided gradients and SGD algorithm.

        Args:
            weightGradients (np.ndarray): calculated weights gradients
            biasGradients (np.ndarray): calculated bias gradients
            weights (np.ndarray): weights of the layer
            biases (np.ndarray): biases of the layer

        Returns:
            tuple: a tuple containing new weights and biases
        """
        weights += self.learning_rate * weight_gradients
        biases += self.learning_rate * bias_gradients
        return (weights, biases)
