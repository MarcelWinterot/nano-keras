import numpy as np


class Adam:
    def __init__(self, learningRate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7) -> None:
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = np.array([])
        self.v_w = np.array([])
        self.m_b = np.array([])
        self.v_b = np.array([])
        self.t = 0

    def __fillArray(self, arr: np.ndarray, targetShape: tuple, isBiases: bool) -> np.ndarray:
        if not isBiases:
            paddingNeeded = (
                max(0, targetShape[0] - arr.shape[0]), max(0, targetShape[1] - arr.shape[1]))
            result = np.pad(
                arr, ((0, paddingNeeded[0]), (0, paddingNeeded[1])), mode='constant')
        else:
            paddingNeeded = max(0, targetShape[0] - arr.shape[0])
            result = np.pad(arr, ((0, paddingNeeded)), mode='constant')

        return result
        """Function that updates params using provided gradients and Adam algorithm. You can read more about it
        at https://arxiv.org/pdf/1412.6980.pdf

        Args:
            gradients (np.ndarray): Array of [[weight gradients], [bias gradients]]
            params (np.ndarray): Array of [[weights], [biases]]

        Returns:
            np.ndarray: Updated params using provided gradients
        """

    def applyGradients(self, weightGradients: np.ndarray, biasGradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple:
        self.t += 1
        beta1T = self.beta1 ** self.t
        beta2T = self.beta2 ** self.t

        if self.m_w.size == 0:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)

        weightGradientsShape = weightGradients.shape
        self.m_w = self.beta1 * \
            self.__fillArray(self.m_w, weightGradientsShape,
                             False)[:weightGradientsShape[0], :weightGradientsShape[1]] + (1 - self.beta1) * weightGradients
        self.v_w = self.beta2 * \
            self.__fillArray(self.v_w, weightGradientsShape,
                             False)[:weightGradientsShape[0], :weightGradientsShape[1]] + (1 - self.beta2) * weightGradients ** 2

        m_hat_w = self.m_w / (1 - beta1T)
        v_hat_w = self.v_w / (1 - beta2T)

        weightsShape = weights.shape

        weights += self.learningRate * \
            m_hat_w[:weightsShape[0], :weightsShape[1]] / \
            (np.sqrt(v_hat_w[:weightsShape[0],
                             :weightsShape[1]]) + self.epsilon)

        biasGradients = biasGradients.flatten()

        biasGradientsShape = biasGradients.shape
        self.m_b = self.beta1 * \
            self.__fillArray(self.m_b, biasGradientsShape,
                             True)[:biasGradientsShape[0]] + (1 - self.beta1) * biasGradients
        self.v_b = self.beta2 * \
            self.__fillArray(self.v_b, biasGradientsShape,
                             True)[:biasGradientsShape[0]] + (1 - self.beta2) * biasGradients**2

        m_hat_b = self.m_b / (1 - beta1T)
        v_hat_b = self.v_b / (1 - beta2T)

        biases += self.learningRate * \
            np.average(m_hat_b) / \
            (np.sqrt(np.average(v_hat_b)) + self.epsilon)

        return (weights, biases)


class SGD:
    def __init__(self, learningRate: float = 0.001):
        self.learningRate = learningRate

    def applyGradients(self, weightGradients: np.ndarray, biasGradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple:
        weights += self.learningRate * weightGradients
        biases += self.learningRate * biasGradients
        return (weights, biases)
