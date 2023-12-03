import numpy as np
import math
from nano_keras.layers import Layer
from nano_keras.optimizers import Optimizer

class MaxPool1D(Layer):
    def __init__(self, pool_size: int = 2, strides: int = 2, name: str = "MaxPool1D") -> None:
        """Intializer for the MaxPool1D layer

        Args:
            pool_size (int, optional): Size of the pooling window. Defaults to 2.
            strides (int, optional): Step the pool should take. Defaults to 2.
            name (str, optional): Name of the layer. Defaults to MaxPool1D.
        """
        self.pool_size: int = pool_size
        self.strides: int = strides
        self.name: str = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        self.output_shape_value: tuple = math.ceil(
            (input_shape - self.kernel_size + 1) / self.strides)
        return self.output_shape_value

    def __repr__(self) -> str:
        return f"{self.name} (MaxPool1D){' ' * (28 - len(self.name) - 11)}{(None, self.output_shape_value)}{' ' * (26-len(f'(None, {self.output_shape_value})'))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Call function for the MaxPool1D layer. It reduces the size of an array by how much the kernel_size and strides is set to.
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
        self.inputs: np.ndarray = x
        x_shape = x.shape
        height: int = (x_shape[0] - self.pool_size) // self.strides + 1
        self.output: np.ndarray = np.ndarray(height)

        self.mask = np.zeros_like(self.inputs)

        if len(x_shape) == 2:
            self.output = np.ndarray((height, x_shape[-1]))

        for i in range(height):
            i_start = i * self.strides
            i_end = i_start + self.pool_size
            index = np.argmax(x[i_start:i_end])

            self.mask[index] = 1

            self.output[i] = x[index]

        return self.output

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        delta = np.average(gradient * self.output)
        return np.dot(delta, self.inputs)


class MaxPool2D(Layer):
    def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] = (2, 2), name: str = "MaxPool2D"):
        """Intializer for the MaxPool2D layer

        Args:
            pool_size (tuple[int, int], optional): Size of the pool. Defaults to (2, 2).
            strides (tuple[int, int], optional): Step the kernel should take. Defaults to (2, 2).
            name (str, optional): Name of the layer. Default to NaxPool2D
        """
        self.pool_size: tuple = pool_size
        self.strides: tuple = strides
        self.name: tuple = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        self.output_shape_value: tuple = tuple([math.floor(
            (input_shape[i] - self.pool_size[i]) / self.strides[i]) + 1 for i in range(2)])

        if len(input_shape) > 2:
            self.output_shape_value += (input_shape[-1],)

        return self.output_shape_value

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'
        return f"{self.name} (MaxPool2D){' ' * (28 - len(self.name) - 11)}{formatted_output}{' ' * (26-len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Call function for the MaxPool2D layer. It reduces the size of an array by how much the kernel_size and strides is set to.
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
        self.inputs: np.ndarray = x

        x_shape = x.shape
        height = (x_shape[0] - self.pool_size[0]) // self.strides[0] + 1
        width = (x_shape[1] - self.pool_size[1]) // self.strides[1] + 1

        self.output = np.zeros((height, width))

        self.mask: np.ndarray = np.zeros_like(self.inputs)

        if len(x_shape) == 3:
            self.output = np.zeros((height, width, x_shape[-1]))
            self.mask: np.ndarray = np.zeros_like(self.inputs)

        for i in range(height):
            for j in range(width):
                # We are using this method instead of range(0, x.shape[0], self.strides[0]) as then we'd have to
                # keep on what iteration of what we are on and that's just too much useless work for now
                i_start, j_start = i * self.strides[0], j * self.strides[1]
                i_end, j_end = i_start + \
                    self.pool_size[0], j_start + self.pool_size[1]

                subarray = x[i_start:i_end, j_start:j_end]
                index = np.argmax(subarray)

                index = np.unravel_index(index, subarray.shape)

                self.mask[index[0], index[1]] = 1

                self.output[i, j] = x[index]

        if len(x.shape) == 3:
            self.output[-1, -1, :] = x[-1, -1, :]

        return self.output

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagation algorithm for MaxPool2D layer. Note that it's not finished so it doesn't work properly

        Args:
            gradient (np.ndarray): Gradient calculated by loss.compute_derivative() or previous layers output gradient
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient
        """
        delta = np.average(gradient * self.output)
        return np.dot(delta, self.inputs)
