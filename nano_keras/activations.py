import numpy as np


class Activation:
    def __init__(self) -> None:
        """Initalizer for most activation function that don't have any special params like LeakyReLU or ELU
        """
        pass

    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Base activation class implementation of compute loss function

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        pass

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Base activation class implementation of the derivative of apply_activation

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them        
        """
        pass


class Sigmoid(Activation):
    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply sigmoid activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return 1 / (1 + np.exp(-X))

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of sigmoid activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        sX = self.apply_activation(X)
        return sX * (1 - sX)


class Tanh(Activation):
    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply tanh activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of tanh activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return 1 - self.apply_activation(X) ** 2


class ReLU(Activation):
    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply ReLU activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.maximum(0.0, X)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of tanh activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.where(X < 0, 0, 1)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.2):
        """Initalizer for LeakyRELU activation function

        Args:
            alpha (float, optional): By how much do we multiply X if it's is smaller than 0. Defaults to 0.2.
        """
        self.alpha: float = alpha

    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply LeakyReLU activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.maximum(self.alpha*X, X)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of LeakyReLU activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.where(X <= 0, 1, self.alpha)


class ELU(Activation):
    def __init__(self, alpha: float = 0.2):
        """Initalizer for ELU activation function

        Args:
            alpha (float, optional): By how much do we multiply np.exp(X) if X is smaller than 0. Defaults to 0.2.
        """
        self.alpha: float = alpha

    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply ELU activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.where(X > 0, X, self.alpha * np.exp(X) - 1)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of ELU activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        return np.where(X >= 0, 1, self.alpha * np.exp(X))


class Softmax(Activation):
    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        """Function to apply softmax activation on given data

        Args:
            X (np.ndarray): Data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        ex = np.exp(X - np.max(X))
        return ex / ex.sum(axis=0)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """Function to apply derivative of softmax activation on given data

        Args:
            X (np.ndarray): data to apply activation function on

        Returns:
            np.ndarray: Data with activation function applied to them
        """
        s = self.apply_activation(X)
        return np.diag(s) - np.outer(s, s)


ACTIVATIONS = {'sigmoid': Sigmoid(), 'tanh': Tanh(), 'relu': ReLU(
), 'leaky_relu': LeakyReLU(), 'elu': ELU(), "softmax": Softmax()}
