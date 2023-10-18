import numpy as np
import math


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def ReLU(x: np.ndarray) -> np.ndarray:
    return np.array([max(0.0, X) for X in x])


def leakyReLU(x: np.ndarray, a: float = 0.2) -> np.ndarray:
    return np.array([X if X > 0 else a*X for X in x])


def ELU(x: np.ndarray, a: float = 0.2) -> np.ndarray:
    return np.array([X if X > 0 else a * (math.exp(X) - 1) for X in x])


def dSigmoid(x: np.ndarray) -> np.ndarray:
    return np.array([sigmoid(X) * (1 - sigmoid(X)) for X in x])


def dTanh(x: np.ndarray) -> np.ndarray:
    return 1 - tanh(x) ** 2


def dReLU(x: np.ndarray) -> np.ndarray:
    return np.array([0.0 if X < 0 else 1 for X in x])


def dLeakyReLU(x: np.ndarray, a: float = 0.2) -> np.ndarray:
    return np.array([1 if X >= 0 else a for X in x])


def dELU(x: np.ndarray, a: float = 0.2) -> np.ndarray:
    return np.array([1 if X >= 0 else a*math.exp(X) for X in x])


if __name__ == "__main__":
    X = np.array([0.17, -0.26, 1])
    print(dSigmoid(X))
    print(dTanh(X))
    print(dReLU(X))
    print(dLeakyReLU(X))
    print(dELU(X))
