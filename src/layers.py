from activations import *
from typing import Union
from activations import Activation, np
from optimizers import Optimizer
from regulizers import Regularizer
import numpy as np
import math

# Currently working on: Conv1D


class Layer:
    def __init__(self, units: int, activation: Union[Activation, str] = None, regulizer: Regularizer = None, name: str = "Layer") -> None:
        """Intializer for the layer class. 

        Args:
            units (int): Number of neurons the layer should have
            activation (Union[Activation, str], optional): Activation function the model should use. You can find them all in the activations.py. Defaults to None, but you should only set it to None if it's the input layer. As otherwise it'd throw an error.
            regulizer (Regularizer, optional): Regulizer the model should use. You can find them all in the regulizers.py file. You must pass the already intialized class. Defaults to None.
            name (str, optional): Name of the layer. Helpful for debugging. Defaults to "Layer".
        """
        activations_ = {'sigmoid': Sigmoid(), 'tanh': Tanh(
        ), 'relu': ReLU(), 'leaky_relu': LeakyReLU(), 'elu': ELU()}
        self.units = units
        self.name = name
        self.weights = np.array([])
        self.biases = np.random.randn(units)
        self.activation = activations_[activation] if type(
            activation) == str else activation
        self.regulizer = regulizer
        # We set the type to dense as every other layer will need it's special init where we'll set it
        self.type = Dense

    def __repr__(self) -> str:
        return "Base layer class"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        weighted_sum = np.dot(x, self.weights) + self.biases
        output = self.activation.compute_loss(weighted_sum)
        self.outputs = np.array([output, weighted_sum])
        return output


class Dense(Layer):
    def __repr__(self) -> str:
        return f"Dense layer: {self.units} units"

    def backpropagate(self, loss: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        """Backpropagation algorithm for the dense layer

        Args:
            loss (np.ndarray): loss calculated by the activation function derivative or loss calculated by previous layers backpropagation algorithm
            optimizer (Optimizer): Optimizer to update the model's parameters

        Returns:
            np.ndarray: new loss
        """
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
        self.type = Dropout

    def __repr__(self) -> str:
        return f"Dropout layer: {self.units} units"

    def __call__(self, x: np.ndarray, isTraining: bool = True) -> np.ndarray:
        self.inputs = x
        if isTraining:
            weighted_sum = np.dot(x, self.weights) + self.biases
            weighted_sum /= 1 - self.dropout_rate
            output = self.activation.compute_loss(weighted_sum)
            self.outputs = np.array([output, weighted_sum])
            return output

        return super().__call__(x)

    def backpropagate(self, loss: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        """Backpropagation algorithm for the dense layer

        Args:
            loss (np.ndarray): loss calculated by the activation function derivative or loss calculated by previous layers backpropagation algorithm 
            optimizer (Optimizer): Optimizer to update the model's parameters

        Returns:
            np.ndarray: new loss
        """
        if self.regulizer is not None:
            loss = self.regulizer.compute_loss(loss, self.weights, self.biases)
        delta = np.average([loss * self.activation.compute_derivative(self.outputs[i])
                           for i in range(len(self.outputs))])
        delta /= 1/(1-self.dropout_rate)  # Scaling the gradient
        weights_gradients = np.outer(self.inputs, delta)
        self.weights, self.biases = optimizer.apply_gradients(
            weights_gradients, np.array(delta, dtype=float), self.weights, self.biases)
        return np.dot(delta, self.weights.T)


class Flatten(Layer):
    def __init__(self) -> None:
        self.type = Flatten

    def __repr__(self) -> str:
        return f"Flatten layer"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.ravel(x)

    def backpropagate(self, loss: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        return loss


class Reshape(Layer):
    def __init__(self, target_shape: tuple) -> None:
        self.target_shape = target_shape
        self.type = Reshape

    def __repr__(self) -> str:
        return f"Reshape layer"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.reshape(x, self.target_shape)

    def backpropagate(self, loss: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        return loss


class MaxPooling1D(Layer):
    def __init__(self, pool_size: int = 2, strides: int = None) -> None:
        """Intializer for the MaxPooling1D layer

        Args:
            pool_size (int, optional): Size of the pooling window. Defaults to 2.
            strides (int, optional): Step the pooling_window should take. If the parameter is set to None it will be assigned the value of pool_size. Defaults to None.
        """
        self.pool_size = pool_size
        self.strides = pool_size if strides is None else strides
        self.type = MaxPooling1D

    def __repr__(self) -> str:
        return f"Max Pooling 1D layer"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call function for the MaxPooling1D layer. It reduces the size of an array by how much the pool_size and strides is set to.
        For example let's say we have those parameters:\n
        array X = [[1., 5., 3., 6., 7., 4.]]\n
        both pool_size and strides set to 2\n
        The result we'd get is [[5., 6., 7.]]\n
        As we take a smaller sub arrays of size pool_size and return the max value out of them, then onto a next sub-array, with index of: current index + strides

        Args:
            x (np.ndarray): Array to reduce the size of

        Returns:
            np.ndarray: Array with reduced size
        """
        output_size = math.ceil((x.size - self.pool_size + 1) / self.strides)
        output = np.empty((output_size,))

        currentIndex = 0
        for i in range(0, x.size, self.strides):
            if i + self.pool_size > x.size:
                break  # Reached the end of the input

            output[currentIndex] = np.max(x[i:i + self.pool_size])
            currentIndex += 1

        return output


class MaxPooling2D(Layer):
    def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] = None):
        """Intializer for the MaxPooling2D layer

        Args:
            pool_size (tuple[int, int], optional): Size of the pooling_window. Defaults to (2, 2).
            strides (tuple[int, int], optional): Step the pooling_window should take. Is the paramter is set to None it will be assigned the value of pool_size. Defaults to None.
        """
        self.pool_size = pool_size
        self.strides = pool_size if strides is None else strides
        self.type = MaxPooling2D

    def __repr__(self) -> str:
        return f"Max Pooling 2D Layer"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call function for the MaxPooling1D layer. It reduces the size of an array by how much the pool_size and strides is set to.
        For example let's say we have those parameters:\n
        array X:\n
        [[2, 3, 5, 9],\n
        [4, 5, 2, 6],\n
        [7, 4, 6, 5],\n
        [8, 3, 4, 1]]\n
        both pool_size and strides set to (2, 2)\n
        The result we'd get is:\n
        [[5, 9],\n
        [8, 6]]\n


        Args:
            x (np.ndarray): Array to reduce the size of

        Returns:
            np.ndarray: Array with reduced size
        """
        h, w = x.shape
        h_out = (h - self.pool_size[0]) // self.strides[0] + 1
        w_out = (w - self.pool_size[1]) // self.strides[1] + 1

        output = np.zeros((h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                i_start, j_start = i * self.strides[0], j * self.strides[1]
                i_end, j_end = i_start + \
                    self.pool_size[0], j_start + self.pool_size[1]
                output[i, j] = np.max(x[i_start:i_end, j_start:j_end])

        return output


if __name__ == "__main__":
    x = np.random.rand(25)
    print(x)
    pool_size = 2
    strides = 1
    layer = MaxPooling1D(pool_size, strides)
    print(layer(x))
    print(len(layer(x)))
