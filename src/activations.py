import numpy as np
import math


class sigmoid:
    def __init__(self) -> None:
        pass

    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        sX = self.compute_loss(X)
        return sX * (1 - sX)


class tanh:
    def __init__(self) -> None:
        pass

    def computeLoss(self, X: np.ndarray) -> np.ndarray:
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    def computeDLoss(self, X: np.ndarray) -> np.ndarray:
        return 1 - self.computeLoss(X) ** 2


class ReLU:
    def __init__(self) -> None:
        pass

    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, X)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        return np.where(X < 0, 0, 1)


class LeakyReLU:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        # return np.array([x if x > 0 else self.alpha*x for x in X])
        return np.maximum(0.0, X)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        # return np.array([1 if x >= 0 else self.alpha for x in X])
        return np.where(X <= 0, 1, self.alpha)


class ELU:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        # return np.array([x if x > 0 else self.alpha * (math.exp(x) - 1) for x in X])
        return np.where(X > 0, X, self.alpha * math.exp(X) - 1)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        # return np.array([1 if x >= 0 else self.alpha*math.exp(x) for x in X])
        return np.where(X >= 0, 1, self.alpha * np.exp(X))


class softmax:
    def __init__(self) -> None:
        pass

    def compute_loss(self, X: np.ndarray) -> np.ndarray:
        ex = np.exp(X - np.max(X))
        return ex / ex.sum(axis=0)

    def compute_derivative(self, X: np.ndarray) -> np.ndarray:
        s = self.compute_loss(X)
        return np.diag(s) - np.outer(s, s)


if __name__ == "__main__":
    X = np.array([0.20666447, 0.983, 0.1])
    loss = softmax()
    print(loss.compute_loss(X))
    print(loss.compute_derivative(X))
