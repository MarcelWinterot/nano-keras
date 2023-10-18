import numpy as np


class L1:
    def __init__(self, strength: float = 1e-3) -> None:
        """
        Loss += λ * Σ|θ_i| \n
        Parameters:
            strength (float): The regularization strength (λ).
        """
        self.strength = strength

    def computeLoss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        loss += self.strength * np.sum(np.abs(weights)) \
            + self.strength * np.sum(np.abs(biases))
        return loss


class L2:
    def __init__(self, strength: float = 1e-3) -> None:
        """
        Loss += λ * Σ(θ_i^2) \n
        Parameters:
            strength (float): The regularization strength (λ).
        """
        self.strength = strength

    def computeLoss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        loss += self.strength * np.sum(np.square(weights)) \
            + self.strength * np.sum(np.square(biases))
        return loss


class L1L2:
    def __init__(self, l1Strength: float = 1e-3, l2Strength: float = 1e-3) -> None:
        """ Combination of L1 and L2 Regulizers \n
        Loss += λ * Σ|θ_i| + λ * Σ(θ_i^2)

        Args:
            l1Strength (float, optional): Regulization strength for L1 reguilzer calculation. Defaults to 1e-3.
            l2Strength (float, optional): Regulization strength for L2 reguilzer calculation. Defaults to 1e-3.
        """
        self.l1Strength = l1Strength
        self.l2Strength = l2Strength

    def computeLoss(self, loss: float, weights: np.ndarray, biases: np.ndarray) -> float:
        loss += self.l1Strength * np.sum(np.abs(weights)) \
            + self.l1Strength * np.sum(np.abs(biases)) \
            + self.l2Strength * np.sum(np.square(weights)) \
            + self.l2Strength * np.sum(np.square(biases))

        return loss


if __name__ == "__main__":
    loss = 0.73812
    weights = np.array([[0.13892], [-1.2731], [0.91539], [1.4292], [-0.09814]])
    bias = np.array([[0.2193], [0.429], [-0.834]])
    l1 = L1()
    print(l1.computeLoss(loss, weights, bias))
    l2 = L2()
    print(l2.computeLoss(loss, weights, bias))
    l1l2 = L1L2()
    print(l1l2.computeLoss(loss, weights, bias))
