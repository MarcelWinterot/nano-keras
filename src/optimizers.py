import numpy as np


class Optimizer:
    def __init__(self) -> None:
        pass

    def fill_array_(self, arr: np.ndarray, target_shape: tuple, isBiases: bool = False) -> np.ndarray:
        """Support function to fill the m_w, v_w and m_b, v_b arrays if their shapes are smaller than the gradients

        Args:
            arr (np.ndarray): array to fill
            target_shape (tuple): what shape should the array have after filling it
            isBiases (bool, optional): If we expand the m_b and v_b we set it to True as bias gradients are 1D. Defaults to False.

        Returns:
            np.ndarray: filled array
        """
        arr_shape = arr.shape
        if not isBiases:
            paddingNeeded = (
                max(0, target_shape[0] - arr_shape[0]), max(0, target_shape[1] - arr_shape[1]))
            result = np.pad(
                arr, ((0, paddingNeeded[0]), (0, paddingNeeded[1])), mode='constant')
        else:
            paddingNeeded = max(0, target_shape[0] - arr_shape[0])
            result = np.pad(arr, ((0, paddingNeeded)), mode='constant')

        return result

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        """Initializer for the SGD(Stochastic Gradient Descent) optimizer

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


class Adam(Optimizer):
    def __init__(self, learningRate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7) -> None:
        """Intializer to the Adam(Adaptive Moment Estimator) optimizer.

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

        # Updating weights
        target_shape = weights.shape

        # Adjusting shapes before calculations
        self.m_w = self.fill_array_(self.m_w, target_shape)[
            :weightGradients.size]
        self.v_w = self.fill_array_(self.v_w, target_shape)[
            :weightGradients.size]

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weightGradients
        self.v_w = self.beta2 * self.v_w + \
            (1 - self.beta2) * weightGradients ** 2

        m_hat_w = self.m_w / (1 - beta1T)
        v_hat_w = self.v_w / (1 - beta2T)

        weights += self.learningRate * \
            m_hat_w[:target_shape[0], :target_shape[1]] / \
            (np.sqrt(v_hat_w[:target_shape[0],
             :target_shape[1]]) + self.epsilon)

        # Updating biases
        target_shape = biases.shape

        # Adjusting shapes before calculations
        self.m_b = self.fill_array_(self.m_b, target_shape, True)[
            :biasGradients.size]
        self.v_b = self.fill_array_(self.v_b, target_shape, True)[
            :biasGradients.size]

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * biasGradients
        self.v_b = self.beta2 * self.v_b + \
            (1 - self.beta2) * biasGradients ** 2

        m_hat_b = self.m_b / (1 - beta1T)
        v_hat_b = self.v_b / (1 - beta2T)

        biases += self.learningRate * m_hat_b / \
            (np.sqrt(v_hat_b) + self.epsilon)

        return (weights, biases)


class Adagrad(Optimizer):
    def __init__(self, learning_rate: float = 0.001) -> None:
        """Intializer for the Adagrad(Adaptive Gradient Descent) optimizer.

        Args:
            learning_rate (float, optional): Paramater that specifies how fast the model will learn. Defaults to 0.001.
        """
        self.learning_rate = learning_rate
        self.e = 1e-7
        self.v_w = None
        self.v_b = None

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
        if self.v_w is None:
            self.v_w = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)

        # Adjusting the shape before calculation
        target_shape = weights.shape
        self.v_w = self.fill_array_(self.v_w, target_shape)[
            :target_shape[0], :target_shape[1]]

        self.v_w += weights_gradients ** 2
        self.v_b += bias_gradients ** 2

        weights += (self.learning_rate /
                    (np.sqrt(self.v_w) + self.e)) * weights_gradients
        biases += (self.learning_rate /
                   (np.sqrt(self.v_b) + self.e)) * bias_gradients

        return (weights, biases)


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9) -> None:
        """Initalizer for the RMSProp(Root Mean Square Propagation) algorithm.

        Args:
            learning_rate (float, optional): Paramter that specifies how fast the model will learn. Defaults to 0.001.
            rho (float, optional): Paramter that controls the exponential moving average of the squared gradients. Defaults to 0.9.
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.e = 1e-7
        self.v_w = None
        self.v_b = None

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
        if self.v_w is None:
            self.v_w = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)

        target_shape = weights.shape

        # Adjusting the shapes before calculation
        self.v_w = self.fill_array_(self.v_w, target_shape)[
            :target_shape[0], :target_shape[1]]

        self.v_w = self.rho * self.v_w + (1 - self.rho) * weights_gradients**2
        self.v_b = self.rho * self.v_b + (1 - self.rho) * bias_gradients**2

        weights += self.learning_rate * \
            (weights_gradients/(np.sqrt(self.v_w) + self.e))
        biases += self.learning_rate * \
            (bias_gradients / (np.sqrt(self.v_b) + self.e))

        return (weights, biases)


class Adadelta(Optimizer):
    def __init__(self, rho: float = 0.9) -> None:
        """Initalizer for the Adadelta(Adaptive delta) algorithm.

        Args:
            rho (float, optional): Parameter that controls an exponential moving average. Defaults to 0.9.
        """
        self.e = 1e-7
        self.rho = rho
        self.v_w = None
        self.v_b = None
        self.v_w_a = None
        self.v_b_a = None

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
        if self.v_w is None:
            self.v_w = np.zeros_like(weights)
            self.v_w_a = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)
            self.v_b_a = np.zeros_like(biases)

        target_shape = weights_gradients.shape

        # Adjusting the shapes before calculation
        self.v_w = self.fill_array_(self.v_w, target_shape)[
            :target_shape[0], :target_shape[1]]
        self.v_w_a = self.fill_array_(self.v_w_a, target_shape)[
            :target_shape[0], :target_shape[1]]

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
    def __init__(self) -> None:
        super().__init__()

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

        return (weights, biases)
