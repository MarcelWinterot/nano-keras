from activations import *
from typing import Union
from activations import Activation, np
from optimizers import Optimizer
from regulizers import Regularizer

"""
TODO 26.10.2023
Try to convert the feed_forward to have the variables stored in self as it might make the code cleaner
Start implementing flatten and reshape layers
"""


class Layer:
    def __init__(self, units: int, activation: Union[Activation, str], regulizer: Union[Regularizer, None] = None, name: str = "Layer") -> None:
        __activations__ = {'sigmoid': Sigmoid(), 'tanh': Tanh(
        ), 'relu': ReLU(), 'leaky_relu': LeakyReLU(), 'elu': ELU()}
        self.units = units
        self.name = name
        self.weights = np.array([])
        self.biases = np.random.randn(units)
        self.activation = __activations__[activation] if type(
            activation) == str else activation
        self.regulizer = regulizer

    def __repr__(self) -> str:
        return "Base layer class"

    def feed_forward(self, x: np.ndarray) -> tuple:
        weighted_sum = np.dot(x, self.weights) + self.biases
        output = self.activation.compute_loss(weighted_sum)
        return output, weighted_sum


class Dense(Layer):
    def __repr__(self) -> str:
        return f"Dense layer: {self.units} units"

    def backpropagate(self, loss: np.ndarray, outputs: np.ndarray, inputs: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        delta = np.average([
            loss * self.activation.compute_derivative(outputs[i]) for i in range(len(outputs))])
        weights_gradients = np.outer(inputs, delta)
        weights, biases = optimizer.apply_gradients(weights_gradients, np.array(
            [delta], dtype=float), self.weights, self.biases)
        self.weights = weights
        self.biases = biases
        loss = np.dot(delta, self.weights.T)
        return loss


class Dropout(Layer):
    def __init__(self, units: int, activation: Union[Activation, str], dropout_rate: float = 0.2, regulizer: Union[Regularizer, None] = None, name: str = "Layer") -> None:
        super().__init__(units, activation, regulizer, name)
        self.dropout_rate = dropout_rate

    def __repr__(self) -> str:
        return f"Dropout layer: {self.units} units"

    def feed_forward(self, x: np.ndarray, isTraining: bool = True) -> tuple:
        if isTraining:
            self.rnd = np.int8(np.random.uniform(
                0., 1., x.shape) > self.dropout_rate)
            weighted_sum = np.dot(x, self.weights) * self.rnd + self.biasesr
            output = self.activation.compute_loss(weighted_sum)
            return weighted_sum, output

        return super().feed_forward(x)


if __name__ == "__main__":
    layer = Dropout(4, "relu", 0.2)
    layer.weights = np.array([0.2, 1, 1, 1])
    layer.biases = np.array([1])
    data = np.array([0.3, 0.6, -0.4, 2])
    print(layer.feed_forward(data))
