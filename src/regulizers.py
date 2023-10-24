import numpy as np

class Regularizer:
    def __init__(self, strength: float = 1e-3):
        self.strength = strength

    def compute_loss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        pass

class L1(Regularizer):
    def compute_loss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        loss += self.strength * np.sum(np.abs(weights)) + self.strength * np.sum(np.abs(biases))
        return loss

class L2(Regularizer):
    def compute_loss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        loss += self.strength * np.sum(np.square(weights)) + self.strength * np.sum(np.square(biases))
        return loss

class L1L2(Regularizer):
    def __init__(self, l1_strength: float = 1e-3, l2_strength: float = 1e-3):
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

    def compute_loss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        loss += self.l1_strength * np.sum(np.abs(weights)) + self.l1_strength * np.sum(np.abs(biases)) + self.l2_strength * np.sum(np.square(weights)) + self.l2_strength * np.sum(np.square(biases))
        return loss

if __name__ == "__main__":
    loss = 0.73812
    weights = np.array([[0.13892], [-1.2731], [0.91539], [1.4292], [-0.09814]])
    bias = np.array([[0.2193], [0.429], [-0.834]])
    
    l1 = L1()
    print(l1.compute_loss(loss, weights, bias))
    
    l2 = L2()
    print(l2.compute_loss(loss, weights, bias))
    
    l1l2 = L1L2()
    print(l1l2.compute_loss(loss, weights, bias))
