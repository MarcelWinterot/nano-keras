from activations import *
from activations import np
from optimizers import Optimizer
from regulizers import Regularizer
import numpy as np
import math

ACTIVATIONS = {'sigmoid': Sigmoid(), 'tanh': Tanh(), 'relu': ReLU(
), 'leaky_relu': LeakyReLU(), 'elu': ELU(), "softmax": Softmax()}


class Layer:
    def __init__(self, units: int, activation: Activation | str = None, regulizer: Regularizer = None, name: str = "Layer") -> None:
        """Intializer for the layer class. 

        Args:
            units (int): Number of neurons the layer should have
            activation (Activation | str, optional): Activation function the model should use. You can find them all in the activations.py. Defaults to None, but you should only set it to None if it's the input layer. As otherwise it'd throw an error.
            regulizer (Regularizer, optional): Regulizer the model should use. You can find them all in the regulizers.py file. You must pass the already intialized class. Defaults to None.
            name (str, optional): Name of the layer. Helpful for debugging. Defaults to "Layer".
        """
        self.units = units
        self.name = name
        self.weights = np.array([])
        self.biases = np.random.randn(units)
        self.activation = ACTIVATIONS[activation] if type(
            activation) == str else activation
        self.regulizer = regulizer

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        return

    def __repr__(self) -> str:
        return "Base layer class"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        weighted_sum = np.dot(x, self.weights) + self.biases
        output = self.activation.compute_loss(weighted_sum)
        self.outputs = np.array([output, weighted_sum])
        return output

    def backpropagate(self, loss: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        """Backpropagation algorithm base implementation for all the layers that don't have any parameters to update

        Args:
            loss (np.ndarray): Loss calculated by loss.compute_derivative()
            optimizer (Optimizer): Optimizer to use when updating layers parameters 

        Returns:
            np.ndarray: New loss
        """
        return loss


class Dense(Layer):
    def __init__(self, units: int, activation: Activation | str = None, regulizer: Regularizer = None, name: str = "Layer") -> None:
        super().__init__(units, activation, regulizer, name)
        self.type = Dense

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        """Function to generate the output shape of a layer

        Args:
            layers (list): All layers in a network
            current_layer_index (int): Index of the current layer

        Returns:
            tuple: output shape
        """
        return self.units

    def __repr__(self) -> str:
        return f"{self.name} (Dense){' ' * (28 - len(self.name) - 7)}{(None, self.units)}{' ' * (26 - len(f'(None, {self.units})'))}{len(self.weights)}\n"

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
        delta = np.average([
            loss * self.activation.compute_derivative(self.outputs[i]) for i in range(len(self.outputs))])
        weights_gradients = np.outer(self.inputs, delta)
        self.weights, self.biases = optimizer.apply_gradients(weights_gradients, np.array(
            delta, dtype=float), self.weights, self.biases)
        return np.dot(delta, self.weights.T)


class Dropout(Layer):
    def __init__(self, units: int, activation: Activation | str, dropout_rate: float = 0.2, regulizer: Regularizer | None = None, name: str = "Layer") -> None:
        super().__init__(units, activation, regulizer, name)
        self.dropout_rate = dropout_rate
        self.type = Dropout

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        """Function to generate the output shape of a layer

        Args:
            layers (list): All layers in a network
            current_layer_index (int): Index of the current layer

        Returns:
            tuple: output shape
        """
        return self.units

    def __repr__(self) -> str:
        return f"{self.name} (Dropout){' ' * (28 - len(self.name) - 9)}{(None, self.units)}{' ' * (26 - len(f'(None, {self.units})'))}{len(self.weights)}\n"

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
        delta = np.average([
            loss * self.activation.compute_derivative(self.outputs[i]) for i in range(len(self.outputs))])
        delta /= 1/(1-self.dropout_rate)  # Scaling the gradient
        weights_gradients = np.outer(self.inputs, delta)
        self.weights, self.biases = optimizer.apply_gradients(
            weights_gradients, np.array(delta, dtype=float), self.weights, self.biases)
        return np.dot(delta, self.weights.T)


class Flatten(Layer):
    def __init__(self, name: str = "Flatten") -> None:
        self.type = Flatten
        self.name = name

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        """Function to generate the output shape of a layer

        Args:
            layers (list): All layers in a network
            current_layer_index (int): Index of the current layer

        Returns:
            tuple: output shape
        """
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        self.output_shape_value = (np.prod(np.array(input_shape)))
        return self.output_shape_value

    def __repr__(self) -> str:
        return f"{self.name} (Flatten){' ' * (28 - len(self.name) - 9)}{(None, self.output_shape_value)}{' ' * (26-len(f'(None, {self.output_shape_value})'))}0\n"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.ravel(x)


class Reshape(Layer):
    def __init__(self, target_shape: tuple, name: str = "Reshape") -> None:
        self.target_shape = target_shape
        self.type = Reshape
        self.name = name

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        """Function to generate the output shape of a layer

        Args:
            layers (list): All layers in a network
            current_layer_index (int): Index of the current layer

        Returns:
            tuple: output shape
        """
        return self.target_shape

    def __repr__(self) -> str:
        return f"{self.name} (Reshape){' ' * (28 - len(self.name) - 9)}{(None, self.target_shape)}{' ' * (26-len(f'(None, {self.target_shape})'))}0\n"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.reshape(x, self.target_shape)


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

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        """Function to generate the output shape of a layer

        Args:
            layers (list): All layers in a network
            current_layer_index (int): Index of the current layer

        Returns:
            tuple: output shape
        """
        # output_shape = (input_shape - pool_size + 1) / strides)
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        return math.ceil((input_shape - self.pool_size + 1) / self.strides)

    def __repr__(self) -> str:
        return f"MaxPooling1D layer"

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

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        # math.floor((input_shape - pool_size) / strides) + 1
        # return math.floor((input_shape - self.pool_size) / self.strides) + 1
        output_shape = [math.floor(
            (input_shape[i] - self.pool_size[i]) / self.strides[i]) + 1 for i in range(2)]
        return tuple(output_shape)

    def __repr__(self) -> str:
        return f"MaxPooling2D Layer"

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


class Conv1D(Layer):
    def __init__(self, filters: int, pool_size: int, strides: int = None, activation: Activation | str = None) -> None:
        self.number_of_filters = filters
        self.pool_size = pool_size
        self.strides = pool_size if strides is None else strides
        self.activation = ACTIVATIONS[activation] if type(
            activation) == str else activation

        self.filters = np.random.randn(self.number_of_filters, self.pool_size)

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        return (input_shape // self.strides, self.number_of_filters)

    def __repr__(self) -> str:
        return f"Conv1D Layer"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x

        weighted_sum = np.zeros(
            (x.size // self.strides, self.number_of_filters))

        for i in range(0, x.size, self.strides):
            if i + self.pool_size > x.size:
                break  # Reached the end of the input

            for j, filter in enumerate(self.filters):
                weighted_sum[i //
                             self.strides, j] = np.sum(x[i:i+self.pool_size] * filter)

        # Applying activation function
        output = self.activation.compute_loss(weighted_sum)
        self.outputs = np.array([output, weighted_sum])

        return output


if __name__ == "__main__":
    x = np.random.randn(2, 5)
    firstLayer = Dropout(32, ReLU())
    secondLayer = Reshape((2, 5))
    thirdLayer = MaxPooling2D((2, 2))

    layers = [firstLayer, secondLayer, thirdLayer]
    print(f"First layers output shape: {firstLayer.output_shape(layers, 0)}")
    print(f"Second layer output shape: {secondLayer.output_shape(layers, 1)}")
    print(f"Third layer output shape: {thirdLayer.output_shape(layers, 2)}")
    print(f"Real third layer output shape: {thirdLayer(x).shape}")
