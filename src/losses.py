import numpy as np

log2 = np.log2
ln = np.log


class Loss:
    def __init__(self) -> None:
        pass

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        pass

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        pass

class MAE(Loss):
    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return np.mean(abs(yTrue - yPred))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return yTrue

class MSE(Loss):
    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> float:
        return np.mean(np.square(yTrue - yPred))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return -2 * (yPred - yTrue)

class BCE(Loss):
    def __init__(self) -> None:
        self.e = 1e-18

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return np.negative(np.average((yTrue * log2(yPred + self.e) + (1 - yTrue) * log2(1 - yPred + self.e))))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return np.negative(yTrue / yPred + ((1 - yTrue) / (1 - yPred)))

class CCE(Loss):
    def __init__(self) -> None:
        self.e = 1e-18

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return np.negative(np.sum(yTrue * ln(yPred + self.e)))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return np.negative(yTrue / (yPred + self.e))

class Hinge(Loss):
    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return np.mean(np.maximum(0, 1 - yTrue * yPred))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return -yTrue * (1 - yTrue * yPred > 0).astype(float)

class Huber(Loss):
    def __init__(self, delta: float = 0.5) -> None:
        self.delta = delta

    def compute_loss(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        absolute_error = np.abs(yTrue - yPred)
        return np.mean(np.where(absolute_error <= self.delta, 0.5 * (absolute_error**2), self.delta * (absolute_error - 0.5 * self.delta)))

    def compute_derivative(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        absolute_error = np.abs(yTrue - yPred)
        return np.where(absolute_error <= self.delta,
                        yPred - yTrue, self.delta * np.sign(yPred - yTrue))

if __name__ == "__main__":
    yTrue = np.array([[0], [0], [1], [1]])
    yPred = np.array([[0], [0.5], [0.951], [1]])
    loss = CCE()
    print(loss.compute_loss(yTrue, yPred))