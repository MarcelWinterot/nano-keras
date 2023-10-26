from activations import *
from typing import Union
from activations import np
from optimizers import Optimizer
from regulizers import Regularizer
import numpy as np

"""
TODO 26.10.2023
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
        self.inputs = x
        weighted_sum = np.dot(x, self.weights) + self.biases
        output = self.activation.compute_loss(weighted_sum)
        self.outputs = np.array([output, weighted_sum])
        return output


class Dense(Layer):
    def __repr__(self) -> str:
        return f"Dense layer: {self.units} units"

    def backpropagate(self, loss: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        if self.regulizer is not None:
            loss = self.regulizer.compute_loss(loss, self.weights, self.biases)
        delta = np.average([loss * self.activation.compute_derivative(self.outputs[i])
                           for i in range(len(self.outputs))])
        weights_gradients = np.outer(self.inputs, delta)
        self.weights, self.biases = optimizer.apply_gradients(weights_gradients, np.array(
            delta, dtype=float), self.weights, self.biases)
        return np.dot(delta, self.weights.T)


class Dropout(Layer):
    def __init__(self, units: int, activation: Union[Activation, str], dropout_rate: float = 0.2, regulizer: Union[Regularizer, None] = None, name: str = "Layer") -> None:
        super().__init__(units, activation, regulizer, name)
        self.dropout_rate = dropout_rate

    def __repr__(self) -> str:
        return f"Dropout layer: {self.units} units"

    def feed_forward(self, x: np.ndarray, isTraining: bool = True) -> tuple:
        self.inputs = x
        if isTraining:
            weighted_sum = np.dot(x, self.weights) + self.biases
            weighted_sum /= 1 - self.dropout_rate
            output = self.activation.compute_loss(weighted_sum)
            self.outputs = np.array([output, weighted_sum])
            return output

        return super().feed_forward(x)

    def backpropagate(self, loss: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        if self.regulizer is not None:
            loss = self.regulizer.compute_loss(loss, self.weights, self.biases)
        delta = np.average([loss * self.activation.compute_derivative(self.outputs[i])
                           for i in range(len(self.outputs))])
        delta /= 1/(1-self.dropout_rate)  # Scaling the gradient
        weights_gradients = np.outer(self.inputs, delta)
        self.weights, self.biases = optimizer.apply_gradients(
            weights_gradients, np.array(delta, dtype=float), self.weights, self.biases)
        return np.dot(delta, self.weights.T)
