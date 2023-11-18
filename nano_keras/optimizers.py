import numpy as np


class Optimizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _fill_array(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Support function to fill the m_w, v_w and m_b, v_b arrays if their shapes are smaller than the gradients

        Args:
            arr (np.ndarray): array to fill
            target_shape (tuple): what shape should the array have after filling it

        Returns:
            np.ndarray: filled array
        """
        arr_shape = arr.shape

        # if len(arr_shape) > len(target_shape):
        #     return arr

        padding_needed = [max(0, target - current)
                          for target, current in zip(target_shape, arr_shape)]
        pad_width = [(0, padding) for padding in padding_needed]

        result = np.pad(arr, pad_width, mode='constant')
        return result

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        """Initializer for the SGD(Stochastic Gradient Descent) optimizer

        Args:
            learning_rate (float, optional): Parameter that spiecfies how fast the model should learn. Defaults to 0.001.
        """
        self.learning_rate: float = learning_rate

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


class Adam(Optimizer):
    def __init__(self, learningRate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7, adjust_biases_shape: bool = False) -> None:
        """Intializer to the Adam(Adaptive Moment Estimator) optimizer.

        Args:
            learningRate (float, optional): Paramter that specifies how fast the model will learn. Defaults to 0.001.
            beta1 (float, optional): Paramter that controls the exponential moving average of the first moment of the gradient. Defaults to 0.9.
            beta2 (float, optional): Paramter that contorls the exponential moving average of the second moment of the gradient. Defaults to 0.999.
            epsilon (float, optional): Paramter that ensures we don't divide by 0 and adds numerical stability to learning rate. Defaults to 1e-7.
            adjust_biases_shape (bool, optional): Paramter that controles wheter we adjuts the bias gradients and moving averages for biases shapes. Default to False.
        """
        self.learningRate: float = learningRate
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.e: float = epsilon
        self.adjust_biases_shape: bool = adjust_biases_shape
        self.m_w: np.ndarray = np.array([])
        self.v_w: np.ndarray = np.array([])
        self.m_b: np.ndarray = np.array([])
        self.v_b: np.ndarray = np.array([])
        self.t: int = 0

    def apply_gradients(self, weightGradients: np.ndarray, biasGradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates params using provided gradients and Adam algorithm. You can read more about it
        at https://arxiv.org/pdf/1412.6980.pdf

        Args:
            weights_gradients (np.ndarray): Weight gradients you've calculated
            bias_gradients (np.ndarray): Bias gradients you've calculated
            weights (np.ndarray): Model or layers weights you want to update
            biases (np.ndarray): Model or layers biases you want to update

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

        if self.adjust_biases_shape:
            target_shape = biases.shape
            self.m_b = self._fill_array(self.m_b, target_shape)[
                :target_shape[0]]
            self.v_b = self._fill_array(self.v_b, target_shape)[
                :target_shape[0]]

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weightGradients
        self.v_w = self.beta2 * self.v_w + \
            (1 - self.beta2) * weightGradients ** 2

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * biasGradients
        self.v_b = self.beta2 * self.v_b + \
            (1 - self.beta2) * biasGradients ** 2

        m_hat_w = self.m_w / (1 - beta1T)
        v_hat_w = self.v_w / (1 - beta2T)

        m_hat_b = self.m_b / (1 - beta1T)
        v_hat_b = self.v_b / (1 - beta2T)

        weights += self.learningRate * m_hat_w / (np.sqrt(v_hat_w) + self.e)
        biases += self.learningRate * m_hat_b / (np.sqrt(v_hat_b) + self.e)

        return (weights, biases)


class Adagrad(Optimizer):
    def __init__(self, learning_rate: float = 0.001, epsilon: float = 1e-7, adjust_biases_shape: bool = False) -> None:
        """Intializer for the Adagrad(Adaptive Gradient Descent) optimizer.

        Args:
            learning_rate (float, optional): Paramater that specifies how fast the model will learn. Defaults to 0.001.
            epsilon (float, optional): Paramter that ensures we don't divide by 0 and adds numerical stability to learning rate. Defaults to 1e-7.
            adjust_biases_shape (bool, optional): Paramter that controles wheter we adjuts the bias gradients and moving averages for biases shapes. Default to False.
        """
        self.learning_rate: float = learning_rate
        self.e: float = epsilon
        self.adjust_biases_shape: bool = adjust_biases_shape
        self.v_w: np.ndarray = np.array([])
        self.v_b: np.ndarray = np.array([])

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates models weights and biases using the Adagrad algorithm. 

        Args:
            weights_gradients (np.ndarray): Weight gradients you've calculated
            bias_gradients (np.ndarray): Bias gradients you've calculated
            weights (np.ndarray): Model or layers weights you want to update
            biases (np.ndarray): Model or layers biases you want to update

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated weights and biases. First element are the weights and second are the biases.
        """
        if self.v_w.size == 0:
            self.v_w = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)

        # Adjusting the shape before calculation
        target_shape = weights.shape

        slices = [slice(0, shape) for shape in target_shape]

        self.v_w = self._fill_array(self.v_w, target_shape)[tuple(slices)]

        if self.adjust_biases_shape:
            target_shape = biases.shape
            self.v_b = self._fill_array(self.v_b, target_shape)[
                :target_shape[0]]

        self.v_w += weights_gradients ** 2
        self.v_b += bias_gradients ** 2

        weights += (self.learning_rate /
                    (np.sqrt(self.v_w) + self.e)) * weights_gradients
        biases += (self.learning_rate /
                   (np.sqrt(self.v_b) + self.e)) * bias_gradients

        return (weights, biases)


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

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates models weights and biases using the RMSprop algorithm. 

        Args:
            weights_gradients (np.ndarray): Weight gradients you've calculated
            bias_gradients (np.ndarray): Bias gradients you've calculated
            weights (np.ndarray): Model or layers weights you want to update
            biases (np.ndarray): Model or layers biases you want to update

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated weights and biases. First element are the weights and second are the biases.
        """
        if self.v_w.size == 0:
            self.v_w = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)

        target_shape = weights.shape

        slices = [slice(0, shape) for shape in target_shape]

        self.v_w = self._fill_array(self.v_w, target_shape)[tuple(slices)]

        if self.adjust_biases_shape:
            target_shape = biases.shape
            self.v_b = self._fill_array(self.v_b, target_shape)[
                :target_shape[0]]

        self.v_w = self.rho * self.v_w + (1 - self.rho) * weights_gradients**2
        self.v_b = self.rho * self.v_b + (1 - self.rho) * bias_gradients**2

        weights += self.learning_rate * \
            (weights_gradients/(np.sqrt(self.v_w) + self.e))
        biases += self.learning_rate * \
            (bias_gradients / (np.sqrt(self.v_b) + self.e))

        return (weights, biases)


class Adadelta(Optimizer):
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

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates models weights and biases using the Adadelta algorithm. 

        Args:
            weights_gradients (np.ndarray): Weight gradients you've calculated
            bias_gradients (np.ndarray): Bias gradients you've calculated
            weights (np.ndarray): Model or layers weights you want to update
            biases (np.ndarray): Model or layers biases you want to update

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

        if self.adjust_biases_shape:
            target_shape = biases.shape
            self.v_b = self._fill_array(self.v_b, target_shape)[
                :target_shape[0]]
            self.v_b_a = self._fill_array(self.v_b_a, target_shape)[
                :target_shape[0]]

        self.v_w = self.rho * self.v_w + \
            (1 - self.rho) * weights_gradients ** 2
        self.v_b = self.rho * self.v_b + (1 - self.rho) + bias_gradients**2

        # It should be -np.sqrt() but I've found that it works better without the minus probably because I'm calculating the gradient incorrectly
        weights_update = np.sqrt(self.v_w_a + self.e) / \
            np.sqrt(self.v_w + self.e) * weights_gradients
        bias_update = np.sqrt(self.v_b_a + self.e) / \
            np.sqrt(self.v_b + self.e) * bias_gradients

        weights += weights_update
        biases += bias_update

        self.v_w_a = self.rho * self.v_w_a + \
            (1 - self.rho) * weights_update ** 2
        self.v_b_a = self.rho * self.v_b_a + (1 - self.rho) * bias_update ** 2

        return (weights, biases)


class NAdam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7, adjust_biases_shape: bool = False) -> None:
        """Initializer for the NAdam(Nesterov-accelerated Adaptive Moment Estimator) algorithm

        Args:
            learningRate (float, optional): Paramter that specifies how fast the model will learn. Defaults to 0.001.
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

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Function that updates models weights and biases using the Adadelta algorithm.

        Args:
            weights_gradients (np.ndarray): Weight gradients you've calculated
            bias_gradients (np.ndarray): Bias gradients you've calculated
            weights (np.ndarray): Model or layers weights you want to update
            biases (np.ndarray): Model or layers biases you want to update

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated weights and biases. First element are the weights and second are the biases.
        """
        if self.m_w.size == 0:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)

        self.t += 1

        # Adjusting the shapes before calculations
        target_shape = weights.shape

        slices = [slice(0, shape) for shape in target_shape]

        self.m_w = self._fill_array(self.m_w, target_shape)[tuple(slices)]
        self.v_w = self._fill_array(self.v_w, target_shape)[tuple(slices)]

        if self.adjust_biases_shape:
            target_shape = biases.shape
            self.m_b = self._fill_array(self.m_b, target_shape)[
                :target_shape[0]]
            self.v_b = self._fill_array(self.v_b, target_shape)[
                :target_shape[0]]

        # Calculations
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_gradients
        self.v_w = self.beta2 * self.v_w + \
            (1 - self.beta2) * weights_gradients ** 2

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * bias_gradients
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * bias_gradients**2

        beta1T = self.beta1 ** self.t
        beta2T = self.beta2 ** self.t

        m_hat_w = (self.beta1 * self.m_w / (1 - beta1T)) + \
            ((1 - self.beta1) * weights_gradients / (1 - beta1T))
        v_hat_w = self.beta2 * self.v_w / (1 - beta2T)

        m_hat_b = (self.beta1 * self.m_b / (1 - beta1T)) + \
            ((1 - self.beta1) * bias_gradients / (1 - beta1T))
        v_hat_b = self.beta2 * self.v_b / (1 - beta2T)

        weights += self.learning_rate / np.sqrt(v_hat_w + self.e) * m_hat_w
        biases += self.learning_rate / np.sqrt(v_hat_b + self.e) * m_hat_b

        return (weights, biases)


OPTIMIZERS = {"adam": Adam(), "sgd": SGD(), "adagrad": Adagrad(
), "adadelta": Adadelta(), "rmsprop": RMSProp(), "nadam": NAdam()}
