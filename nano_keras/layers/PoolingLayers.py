import numpy as np
from numpy.lib.stride_tricks import as_strided
import math
from nano_keras.layers import Layer
from nano_keras.optimizers import Optimizer


class PoolingLayey1D(Layer):
    """Pooling layer 1d base class. It's used to reduce the size of 2d arrays
    """

    def __init__(self, pool_size: int = 2, strides: int = 2, input_shape: tuple = None, name: str = "Pool1D") -> None:
        """Intializer for the PoolingLayer1D layers

        Args:
            pool_size (int, optional): Size of the pooling window. Defaults to 2.
            strides (int, optional): Step the pool should take. Defaults to 2.
            input_shape (tuple, optional): Input shape to the layer. Used if you dont't want to use Input layer. If it's None it won't be used. Defaults to None.
            name (str, optional): Name of the layer. Defaults to Pool1D.
        """
        self.pool_size: int = pool_size
        self.strides: int = strides
        self.input_shape: tuple = input_shape
        self.name: str = name
        self.current_batch: int = 0

    def set_batch_size(self, batch_size: int, layers: list, index: int) -> None:
        self.batch_size: int = batch_size

        input_shape: tuple = layers[index-1].output_shape(layers, index-1)

        self.mask: np.ndarray = np.ndarray((batch_size, *input_shape))

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape: tuple = layers[current_layer_index -
                                    1].output_shape(layers, current_layer_index-1) if self.input_shape is None else self.input_shape
        self.output_shape_value: tuple = math.ceil(
            (input_shape - self.kernel_size + 1) / self.strides)
        return self.output_shape_value

    def __repr__(self) -> str:
        return f"{self.name} (Pool1D){' ' * (28 - len(self.name) - 11)}{(None, self.output_shape_value)}{' ' * (26-len(f'(None, {self.output_shape_value})'))}0\n"

    def __call__(self, x: np.ndarray, option: str, is_training: bool = False) -> np.ndarray:
        """Call function for the Pool1D layers. It reduces the size of an array by how much the kernel_size and strides is set to.

        Args:
            x (np.ndarray): Array to reduce the size of
            option (str): Should we use max or min pooling. Options are: "max" and "min"
            is_training (bool): Determines whether the layer should behave like in the training loop or no. Defaults to False.

        Returns:
            np.ndarray: Array with reduced size
        """
        self.inputs: np.ndarray = x

        x_shape: tuple = x.shape
        height: int = (x_shape[0] - self.pool_size[0]) // self.strides[0] + 1
        channels: int = x_shape[1]

        mask: np.ndarray = np.zeros_like(self.inputs)
        output: np.ndarray = np.zeros((height, channels))

        for c in range(channels):
            for i in range(height):
                i_start: int = i * self.strides[0]
                i_end: int = i_start + self.pool_size[0]

                subarray: np.ndarray = x[i_start:i_end, c]
                index: np.int_ = np.argmax(
                    subarray) if option == "max" else np.argmin(subarray)

                index = np.unravel_index(index, subarray.shape)

                mask[index[0], c] = 1

                output[i, c] = x[index[0], c]

        if is_training:
            self.mask[self.current_batch] = mask
            self.current_batch += 1

        return output

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        """Backpropagate algorithm used for Pooling1D layers.

        Args:
            gradient (np.ndarray): Gradient calculated by loss.compute_derivative() or previous layers output gradient
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient
        """
        # Note that I haven't tested this code so I don't know if it works
        gradient_shape: tuple = gradient.shape
        gradient_expended: bool = False
        mask = np.average(self.mask, 0)

        if (gradient_shape[0] * 2 < self.inputs.shape[0]):
            gradient_expended = True
            extra_dim: np.ndarray = np.zeros(
                (1, gradient.shape[1]))

            gradient = np.vstack((gradient, extra_dim))

        delta: np.ndarray = np.repeat(gradient, 2, axis=0)

        if gradient_expended:
            delta = delta[0:delta.shape[0]-1, :]

        self.current_batch = 0

        return delta * mask


class PoolingLayey2D(Layer):
    """Pooling layer 2d base class. It's used to reduce the size of 2d arrays 
    """

    def __init__(self, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] = (2, 2), input_shape: tuple = None, name: str = "Pool2D"):
        """Intializer for the Pool2D layers

        Args:
            pool_size (tuple[int, int], optional): Size of the pool. Defaults to (2, 2).
            strides (tuple[int, int], optional): Step the kernel should take. Defaults to (2, 2).
            input_shape (tuple, optional): Input shape to the layer. Used if you dont't want to use Input layer. If it's None it won't be used. Defaults to None.
            name (str, optional): Name of the layer. Default to NaxPool2D
        """
        self.pool_size: tuple = pool_size
        self.strides: tuple = strides
        self.input_shape: tuple = input_shape
        self.name: tuple = name
        self.current_batch: int = 0

    def set_batch_size(self, batch_size: int, layers: list, index: int) -> None:
        self.batch_size: int = batch_size

        input_shape: tuple = layers[index-1].output_shape(
            layers, index-1) if self.input_shape is None else self.input_shape

        self.mask: np.ndarray = np.ndarray((batch_size, *input_shape))

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1) if self.input_shape is None else self.input_shape

        self.output_shape_value: tuple = tuple([math.floor(
            (input_shape[i] - self.pool_size[i]) / self.strides[i]) + 1 for i in range(2)])

        if len(input_shape) > 2:
            self.output_shape_value += (input_shape[-1],)

        return self.output_shape_value

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'
        return f"{self.name} (Pool2D){' ' * (28 - len(self.name) - 11)}{formatted_output}{' ' * (26-len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray, option: str, is_training: bool = False) -> np.ndarray:
        """Call function for the Pool2D layers. It reduces the size of an array by how much the kernel_size and strides is set to.

        Args:
            x (np.ndarray): Array to reduce the size of
            option (str): Should we use max or min pooling. Options are: "max" and "min"
            is_training (bool): Determines whether the layer should behave like in the training loop or no. Defaults to False.

        Returns:
            np.ndarray: Array with reduced size
        """
        shape = x.shape
        height = (shape[0] - self.pool_size[0]) // self.strides[0] + 1
        width = (shape[1] - self.pool_size[1]) // self.strides[1] + 1

        output = np.ndarray((height, width, shape[2]))
        mask = np.zeros_like(x)

        out_shape = ((shape[0] - self.pool_size[0])//self.strides[0] + 1,
                     (shape[1] - self.pool_size[1])//self.strides[1] + 1) + self.pool_size

        out_strides = (self.strides[0]*x.strides[0], self.strides[1]
                       * x.strides[1]) + (x.strides[0], x.strides[1])

        for channel in range(shape[2]):
            subarray = x[:, :, channel]

            x_s = as_strided(subarray, shape=out_shape, strides=out_strides)

            vals = np.max(x_s, axis=(-1, -2)
                          ) if option == "max" else np.min(x_s, axis=(-1, -2))

            output[:, :, channel] = vals

            vals: np.ndarray = np.repeat(
                np.repeat(vals, 2, axis=0), 2, axis=1)

            # Handlign the shape of vals and mask
            if subarray.shape[0] != vals.shape[0]:
                if subarray.shape[0] != mask.shape[0]:
                    subarray = subarray[:-1, :-1]
                else:
                    extra_dim_0: np.ndarray = np.zeros(
                        (1, vals.shape[1]))
                    extra_dim_1: np.ndarray = np.zeros(
                        (vals.shape[0] + 1, 1))

                    vals = np.concatenate(
                        (extra_dim_1, np.concatenate((extra_dim_0, vals), axis=0)), axis=1)

            mask[:, :, channel] = (subarray == vals)

        if is_training:
            self.mask[self.current_batch] = mask
            self.inputs = x
            self.current_batch += 1

        return output

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagate algorithm used for Pooling2D layers.

        Args:
            gradient (np.ndarray): Gradient calculated by loss.compute_derivative() or previous layers output gradient
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient
        """
        gradient_shape: tuple = gradient.shape
        channels: int = gradient_shape[2]
        gradient_expended: bool = False
        mask = np.average(self.mask, 0)

        if (gradient_shape[0] * 2 < self.inputs.shape[0]):
            gradient_expended = True

            extra_dim_0: np.ndarray = np.zeros(
                (1, gradient_shape[1], channels))
            extra_dim_1: np.ndarray = np.zeros(
                (gradient_shape[0] + 1, 1, channels))

            gradient = np.concatenate(
                (extra_dim_1, np.concatenate((extra_dim_0, gradient), axis=0)), axis=1)

        delta: np.ndarray = np.repeat(
            np.repeat(gradient, 2, axis=0), 2, axis=1)

        if gradient_expended:
            d_shape: tuple = delta.shape
            delta = delta[0:d_shape[0]-1, 0:d_shape[1]-1, :]

        self.current_batch = 0

        return delta * mask


class MaxPool1D(PoolingLayey1D):
    """MaxPooling layer 1d class. It's used to reduce the size of 1d arrays, by taking the max value of a kernel sized window
    """

    def __repr__(self) -> str:
        return f"{self.name} (MaxPool1D){' ' * (28 - len(self.name) - 11)}{(None, self.output_shape_value)}{' ' * (26-len(f'(None, {self.output_shape_value})'))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        return super().__call__(x, "max", is_training)


class MaxPool2D(PoolingLayey2D):
    """MaxPooling layer 2d class. It's used to reduce the size of 2d arrays, by taking the max value of a kernel sized window
    """

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'
        return f"{self.name} (MaxPool2D){' ' * (28 - len(self.name) - 11)}{formatted_output}{' ' * (26-len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        return super().__call__(x, "max", is_training)


class MinPool1D(PoolingLayey1D):
    """MinPooling layer 1d class. It's used to reduce the size of 1d arrays, by taking the min value of a kernel sized window
    """

    def __repr__(self) -> str:
        return f"{self.name} (MinPool1D){' ' * (28 - len(self.name) - 11)}{(None, self.output_shape_value)}{' ' * (26-len(f'(None, {self.output_shape_value})'))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        return super().__call__(x, "min", is_training)


class MinPool2D(PoolingLayey2D):
    """MinPooling layer 2d class. It's used to reduce the size of 2d arrays, by taking the min value of a kernel sized window
    """

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'
        return f"{self.name} (MinPool2D){' ' * (28 - len(self.name) - 11)}{formatted_output}{' ' * (26-len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        return super().__call__(x, "min", is_training)


class AvgPool1D(PoolingLayey1D):
    """AvgPooling layer 1d class. It's used to reduce the size of 1d arrays, by taking the avg value of a kernel sized window
    """

    def __repr__(self) -> str:
        return f"{self.name} (AvgPool1D){' ' * (28 - len(self.name) - 11)}{(None, self.output_shape_value)}{' ' * (26-len(f'(None, {self.output_shape_value})'))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.inputs: np.ndarray = x

        x_shape: tuple = x.shape
        height: int = (x_shape[0] - self.pool_size[0]) // self.strides[0] + 1
        channels: int = x_shape[1]

        mask: np.ndarray = np.zeros_like(self.inputs)
        output: np.ndarray = np.zeros((height, channels))

        for c in range(channels):
            for i in range(height):
                i_start: int = i * self.strides[0]
                i_end: int = i_start + self.pool_size[0]

                subarray: np.ndarray = x[i_start:i_end, c]

                subarray_sum: np.float_ = np.sum(subarray)

                mask[i_start:i_end, c] = subarray / \
                    subarray_sum if subarray_sum != 0 else 0

                output[i, c] = subarray_sum / subarray.size

        if is_training:
            self.mask[self.current_batch] = mask
            self.current_batch += 1

        return output


class AvgPool2D(PoolingLayey2D):
    """AvgPooling layer 2d class. It's used to reduce the size of 2d arrays, by taking the avg value of a kernel sized window
    """

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'
        return f"{self.name} (AvgPool2D){' ' * (28 - len(self.name) - 11)}{formatted_output}{' ' * (26-len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.inputs: np.ndarray = x

        x_shape: tuple = x.shape
        height: int = (x_shape[0] - self.pool_size[0]) // self.strides[0] + 1
        width: int = (x_shape[1] - self.pool_size[1]) // self.strides[1] + 1
        channels: int = x_shape[2]

        mask: np.ndarray = np.zeros_like(self.inputs)
        output: np.ndarray = np.zeros((height, width, channels))

        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    i_start, j_start = i * self.strides[0], j * self.strides[1]
                    i_end, j_end = i_start + \
                        self.pool_size[0], j_start + self.pool_size[1]

                    subarray: np.ndarray = x[i_start:i_end, j_start:j_end, c]

                    subarray_sum: np.float_ = np.sum(subarray)

                    mask[i_start:i_end, j_start:j_end,
                         c] = subarray / subarray_sum if subarray_sum != 0 else 0

                    output[i, j, c] = subarray_sum / subarray.size

        if is_training:
            self.mask[self.current_batch] = mask
            self.current_batch += 1

        return output
