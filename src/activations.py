import numpy as np
import math


class Activation:
    def __init__(self) -> None:
        pass

    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        pass

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(Activation):
    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        sX = self.compute_loss(X)
        return sX * (1 - sX)


class Tanh(Activation):
    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        return 1 - self.compute_loss(X) ** 2


class ReLU(Activation):
    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, X)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        return np.where(X < 0, 0, 1)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, X)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        return np.where(X <= 0, 1, self.alpha)


class ELU(Activation):
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, X, self.alpha * np.exp(X) - 1)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        return np.where(X >= 0, 1, self.alpha * np.exp(X))


class Softmax(Activation):
    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        ex = np.exp(X - np.max(X))
        return ex / ex.sum(axis=0)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        s = self.compute_loss(X)
        return np.diag(s) - np.outer(s, s)


if __name__ == "__main__":
    X = np.array([0.20666447, 0.983, 0.1])
    activation = Softmax()
    print(activation.compute_loss(X))
    print(activation.compute_derivative(X))
