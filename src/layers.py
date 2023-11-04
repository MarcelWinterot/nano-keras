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


class Input(Layer):
    def __init__(self, input_shape: tuple, useBiases: bool = True, name: str = "Input") -> None:
        self.input_shape = input_shape
        self.name = name
        self.biases = None
        if useBiases:
            try:
                self.biases = np.random.randn(*input_shape)
            except:
                self.biases = np.random.randn(input_shape)

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        return self.input_shape

    def __repr__(self) -> str:
        return f"{self.name} (Input){' ' * (28 - len(self.name) - 7)}{(None, self.input_shape)}{' ' * (26 - len(f'(None, {self.input_shape})'))}0\n"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.biases is not None:
            x = x + self.biases
        return x

    def backpropagate(self, loss: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        return loss


class Dense(Layer):
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
        return f"{self.name} (Dense){' ' * (28 - len(self.name) - 7)}{(None, self.units)}{' ' * (26 - len(f'(None, {self.units})'))}{self.weights.size + self.biases.size}\n"

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
        return f"{self.name} (Dropout){' ' * (28 - len(self.name) - 9)}{(None, self.units)}{' ' * (26 - len(f'(None, {self.units})'))}{self.weights.size + self.biases.size}\n"

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
        formatted_output = f'(None, {", ".join(map(str, self.target_shape))})'
        return f"{self.name} (Reshape){' ' * (28 - len(self.name) - 9)}{formatted_output}{' ' * (26-len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.reshape(x, self.target_shape)


class MaxPooling1D(Layer):
    def __init__(self, kernel_size: int = 2, strides: int = None, name: str = "MaxPool1D") -> None:
        """Intializer for the MaxPooling1D layer

        Args:
            kernel_size (int, optional): Size of the pooling window. Defaults to 2.
            strides (int, optional): Step the kernel should take. If the parameter is set to None it will be assigned the value of kernel_size. Defaults to None.
        """
        self.kernel_size = kernel_size
        self.strides = kernel_size if strides is None else strides
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
        self.output_shape_value = math.ceil(
            (input_shape - self.kernel_size + 1) / self.strides)
        return self.output_shape_value

    def __repr__(self) -> str:
        return f"{self.name} (MaxPool1D){' ' * (28 - len(self.name) - 11)}{(None, self.output_shape_value)}{' ' * (26-len(f'(None, {self.output_shape_value})'))}0\n"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call function for the MaxPooling1D layer. It reduces the size of an array by how much the kernel_size and strides is set to.
        For example let's say we have those parameters:\n
        array X = [[1., 5., 3., 6., 7., 4.]]\n
        both kernel_size and strides set to 2\n
        The result we'd get is [[5., 6., 7.]]\n
        As we take a smaller sub arrays of size kernel_size and return the max value out of them, then onto a next sub-array, with index of: current index + strides

        Args:
            x (np.ndarray): Array to reduce the size of

        Returns:
            np.ndarray: Array with reduced size
        """
        output_size = math.ceil((x.size - self.kernel_size + 1) / self.strides)
        output = np.empty((output_size,))

        currentIndex = 0
        for i in range(0, x.size, self.strides):
            if i + self.kernel_size > x.size:
                break  # Reached the end of the input

            output[currentIndex] = np.max(x[i:i + self.kernel_size])
            currentIndex += 1

        return output


class MaxPooling2D(Layer):
    def __init__(self, kernel_size: tuple[int, int] = (2, 2), strides: tuple[int, int] = None, name: str = "MaxPool2D"):
        """Intializer for the MaxPooling2D layer

        Args:
            kernel_size (tuple[int, int], optional): Size of the kernel. Defaults to (2, 2).
            strides (tuple[int, int], optional): Step the kernel should take. Is the paramter is set to None it will be assigned the value of kernel_size. Defaults to None.
        """
        self.kernel_size = kernel_size
        self.strides = kernel_size if strides is None else strides
        self.name = name

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        self.output_shape_value = tuple([math.floor(
            (input_shape[i] - self.kernel_size[i]) / self.strides[i]) + 1 for i in range(2)])
        return self.output_shape_value

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'
        return f"{self.name} (MaxPool2D){' ' * (28 - len(self.name) - 11)}{formatted_output}{' ' * (26-len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call function for the MaxPooling1D layer. It reduces the size of an array by how much the kernel_size and strides is set to.
        For example let's say we have those parameters:\n
        array X:\n
        [[2, 3, 5, 9],\n
        [4, 5, 2, 6],\n
        [7, 4, 6, 5],\n
        [8, 3, 4, 1]]\n
        both kernel_size and strides set to (2, 2)\n
        The result we'd get is:\n
        [[5, 9],\n
        [8, 6]]\n


        Args:
            x (np.ndarray): Array to reduce the size of

        Returns:
            np.ndarray: Array with reduced size
        """
        x.shape
        height = (x.shape[0] - self.kernel_size[0]) // self.strides[0] + 1
        width = (x.shape[1] - self.kernel_size[1]) // self.strides[1] + 1

        output = np.zeros((height, width))

        for i in range(0, height, self.strides[0]):
            for j in range(0, width, self.strides[1]):
                if i + self.kernel_size[0] > x.shape[0] or j + self.kernel_size[1] > x.shape[1]:
                    break  # Reached the end of the input
                output[i, j] = np.max(
                    x[i:i+self.kernel_size[0], j:j+self.kernel_size[1]])

        return output


class Conv1D(Layer):
    def __init__(self, filters: int = 1, kernel_size: int = 2, strides: int = 2, activation: Activation | str = "relu", name: str = "Conv1D") -> None:
        self.number_of_filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = ACTIVATIONS[activation] if type(
            activation) == str else activation
        self.weights = np.random.randn(filters, self.kernel_size)
        self.name = name
        self.biases = np.random.randn(filters)

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        self.output_shape_value = input_shape // self.strides, self.number_of_filters
        return self.output_shape_value

    def __repr__(self) -> str:
        return f"{self.name} (Conv1D){' ' * (28 - len(self.name) - 8)}{self.output_shape_value}{' ' * (26-len(f'{self.output_shape_value}'))}{self.weights.size + self.biases.size}\n"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x

        weighted_sum = np.zeros(
            (x.size // self.strides, self.number_of_filters))

        for i in range(0, x.size, self.strides):
            if i + self.kernel_size > x.size:
                break  # Reached the end of the input

            for j in range(len(self.weights)):
                weighted_sum[i //
                             self.strides, j] = np.sum(x[i:i+self.kernel_size, j:j+self.kernel_size] * self.weights[j])

        weighted_sum = weighted_sum + self.biases

        # Applying activation function
        output = self.activation.compute_loss(weighted_sum)
        self.outputs = np.array([output, weighted_sum])

        return output


class Conv2D(Layer):
    def __init__(self, filters: int = 1, kernel_size: tuple = (2, 2), strides: tuple = (2, 2), activation: Activation | str = "relu", name: str = "Conv2D") -> None:
        self.number_of_filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = ACTIVATIONS[activation] if type(
            activation) == str else activation
        self.weights = np.random.randn(*kernel_size, filters)
        self.name = name

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        height = (input_shape[0] - self.kernel_size[0]) // self.strides[0] + 1
        width = (input_shape[1] - self.kernel_size[1]) // self.strides[1] + 1
        channels = self.number_of_filters
        self.output_shape_value = (height, width, channels)
        return self.output_shape_value

    def __repr__(self) -> str:
        return f"{self.name} (Conv2D){' ' * (28 - len(self.name) - 8)}{self.output_shape_value}{' ' * (26-len(f'{self.output_shape_value}'))}{self.weights.size}\n"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x

        height = (x.shape[0] - self.kernel_size[0]) // self.strides[0] + 1
        width = (x.shape[1] - self.kernel_size[1]) // self.strides[1] + 1
        channels = self.number_of_filters
        weighted_sum = np.zeros((height, width, channels))

        for i in range(0, height, self.strides[0]):
            for j in range(0, width, self.strides[1]):
                if i + self.strides[0] > self.kernel_size[0] or j + self.strides[1] > self.kernel_size[1]:
                    break  # Reached the end of the input

                for k in range(channels):
                    weighted_sum[i, j, k] = np.sum(x[i:i + self.kernel_size[0],
                                                     j:j + self.kernel_size[1],
                                                     :] * self.weights[:, :, k])

        output = self.activation.compute_loss(weighted_sum)
        self.outputs = np.array([output, weighted_sum])
        return output


if __name__ == "__main__":
    INPUT_SHAPE = (6, 10, 2)
    x = np.random.randn(*INPUT_SHAPE)
    layer1 = Reshape(INPUT_SHAPE)
    layer = Conv2D(2)

    layers = [layer1, layer]

    output = layer(x)
    print(f"Output: {output}")
    print(f"Predicted output shape: {output.shape}")
    print(f"Correct output shape: {layer.output_shape(layers, 1)}")
