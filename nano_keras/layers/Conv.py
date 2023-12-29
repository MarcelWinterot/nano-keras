import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer
from nano_keras.activations import Activation, ACTIVATIONS
from nano_keras.initializers import Initializer, INITIALIZERS


class Conv1D(LayerWithParams):
    """Conv1D layer class. The input shape is (None, steps, channels) and the output shape is (None, new_steps, filters)
    """

    def __init__(self, filters: int = 1, kernel_size: int = 2, strides: int = 2, activation: Activation | str = "relu", weight_initialization: str = "he", regulizer: Regularizer = None, trainable: bool = True, input_shape: tuple = None, name: str = "Conv1D") -> None:
        """Initalizer for the Conv1D layer

        Args:
            filters (int, optional): Number of filters. Defaults to 1.
            kernel_size (int, optional): Kernel size the model should use. Defaults to 2.
            strides (int, optional): By how much should the kernel move after each operation. Defaults to 2.
            activation (Activation | str, optional): Activation function the layer should use. Defaults to "relu".
            weight_initaliziton (str, optional): Weights intialization strategy you want to use to generate weights of the layer. Your options are: random, xavier, he. Defalut to "he"
            regulizer (Regularizer, optional): Regulizer of the layer. Defaults to None.
            trainable (bool, optional): Parameter that decides whether the parameters should be updated or no. Defaults to True.
            input_shape (tuple, optional): Input shape to the layer. Used if you dont't want to use Input layer. If it's None it won't be used. Defaults to None.
            name (str, optional): Name of the layer. Defaults to "Conv1D".
        """
        self.number_of_filters: int = filters
        self.kernel_size: int = kernel_size
        self.strides: int = strides
        self.activation: Activation = ACTIVATIONS[activation] if type(
            activation) == str else activation
        self.weights: np.ndarray = np.random.randn(filters, self.kernel_size)
        self.biases: np.ndarray = np.random.randn(filters)
        self.weight_initialization = weight_initialization
        self.regulizer: Regularizer = regulizer
        self.trainable: bool = trainable
        self.input_shape = input_shape
        self.name: str = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1) if self.input_shape is None else self.input_shape
        self.output_shape_value: tuple = tuple(
            input_shape).size // self.strides, self.number_of_filters
        return self.output_shape_value

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'
        return f"{self.name} (Conv1D){' ' * (28 - len(self.name) - 8)}{formatted_output}{' ' * (26-len(formatted_output))}{self.weights.size + self.biases.size}\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.inputs: np.ndarray = x

        weighted_sum = np.zeros(
            (x.size // self.strides, self.number_of_filters))

        for i in range(0, x.size, self.strides):
            if i + self.kernel_size > x.size:
                break  # Reached the end of the input

            for j in range(len(self.weights)):
                weighted_sum[i //
                             self.strides, j] = np.sum(x[i:i+self.kernel_size, :] * self.weights[j])

        weighted_sum = weighted_sum + self.biases

        output = self.activation.apply_activation(weighted_sum)
        self.outputs: np.ndarray = np.array([output, weighted_sum])

        return output

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagation algorithm for Conv1D layer

        Args:
            gradient (np.ndarray): Gradient calculated by loss.compute_derivative() or previous layers output gradient
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient
        """
        if self.regulizer:
            gradient = self.regulizer.update_gradient(
                gradient, self.weights, self.biases)

        delta = np.average(
            [gradient * self.activation.compute_derivative(output) for output in self.outputs])

        # weights_gradients = delta * self.weights
        weights_gradients = np.zeros(
            (self.inputs.size // self.strides, self.number_of_filters))

        for i in range(0, self.inputs.size, self.strides):
            if i + self.kernel_size > self.inputs.size:
                break  # Reached the end of the input

            for j in range(len(self.weights)):
                weights_gradients[i //
                                  self.strides, j] = np.sum(self.inputs[i:i+self.kernel_size, j:j+self.kernel_size] * delta)

        if self.trainable:
            self.weights, self.biases = optimizer[0].apply_gradients(weights_gradients, np.array(
                delta, dtype=float), self.weights, self.biases)

        return np.dot(delta, self.weights.T)


class Conv2D(LayerWithParams):
    """Conv2D layer class. The input shape is (None, height, width, channels) and the output shape is (None, new_height, new_width, filters)
    """

    def __init__(self, filters: int = 1, kernel_size: tuple = (2, 2), strides: tuple = (1, 1), activation: Activation | str = "relu", weight_initialization: Initializer | str = "he_normal", bias_initialization: Initializer | str = "zeros", regulizer: Regularizer = None, trainable: bool = True, input_shape: tuple = None, name: str = "Conv2D") -> None:
        """Intializer for the Conv2D layer

        Args:
            filters (int, optional): Amount of filters in the layer. Defaults to 1.
            kernel_size (tuple, optional): Kernel size the layer should use. Defaults to (2, 2).
            strides (tuple, optional): By how much should the kernel move. Defaults to (1, 1).
            activation (Activation | str, optional): Activation function of the layer. Defaults to "relu".
            weight_initialization (str, optional): Weights intialization strategy you want to use to generate weights of the layer. Your options are: random_normal, xavier_normal, he_normal. Defalut to "he_normal"
            bias_initialization (str, optional): Bias intialization strategy you want to use to generate biases of the layer. Your options are: random_normal, xavier_normal, he_normal. Defalut to "random_normal"
            regulizer (Regularizer, optional): Regulizer for the layer. Defaults to None.
            trainable (bool, optional): Parameter that decides whether the parameters should be updated or no. Defaults to True.
            input_shape (tuple, optional): Input shape to the layer. Used if you dont't want to use Input layer. If it's None it won't be used. Defaults to None.
            name (str, optional): Name of the layer. Defaults to "Conv2D".
        """
        self.number_of_filters: int = filters
        self.kernel_size: tuple = kernel_size
        self.strides: tuple = strides
        self.activation: Activation = ACTIVATIONS[activation] if type(
            activation) == str else activation

        self.weight_initialization: Initializer = weight_initialization if type(
            weight_initialization) == Initializer else INITIALIZERS[weight_initialization]
        self.bias_initialization: Initializer = bias_initialization if type(
            bias_initialization) == Initializer else INITIALIZERS[bias_initialization]
        self.trainable: bool = trainable

        self.regulizer: Regularizer = regulizer
        self.input_shape: tuple = input_shape
        self.name: str = name
        self.weights: np.ndarray = np.array([])
        self.biases: np.ndarray = np.array([])

        self.current_batch = 0

    def set_batch_size(self, batch_size: int, layers: list, index: int) -> None:
        self.batch_size = batch_size

        input_shape = layers[index-1].output_shape(
            layers, index-1) if self.input_shape is None else self.input_shape
        output_shape = self.output_shape(layers, index)

        self.inputs = np.ndarray((self.batch_size, *input_shape)) if type(
            input_shape) == tuple else np.ndarray((self.batch_size, input_shape))
        self.outputs = np.ndarray((self.batch_size, *output_shape)) if type(
            output_shape) == tuple else np.ndarray((self.batch_size, output_shape))

        x_col_indices = self.im2col_indices(input_shape)

        self.x_col = np.ndarray((self.batch_size, *x_col_indices[0].shape))

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_, bias_data_type: np.float_) -> None:
        """Function used for weights generation for Conv2D layer with 4d weights. The shape of the weights is (kernel_size[0], kernel_size[1], input_shape[-1], number_of_filters)

        Args:
            layers (list): All layers in the model
            current_layer_index (int): For what layer do we want to generate the weights
            weight_data_type (np.float_): In what data type do you want to store the weights. Only use datatypes like np.float32 and np.float64
            bias_data_type (np.float_): In what data type do you want to store the biases. Only use datatypes like np.float32 and np.float64
        """
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1) if self.input_shape is None else self.input_shape

        weights_shape = (self.kernel_size[0], self.kernel_size[1],
                         input_shape[-1], self.number_of_filters)

        self.weights = self.weight_initialization(
            weights_shape, weight_data_type)

        self.biases = self.bias_initialization(
            self.number_of_filters, bias_data_type)

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        self.input_shape: tuple = layers[current_layer_index -
                                         1].output_shape(layers, current_layer_index-1) if self.input_shape is None else self.input_shape
        height = (self.input_shape[0] -
                  self.kernel_size[0]) // self.strides[0] + 1
        width = (self.input_shape[1] -
                 self.kernel_size[1]) // self.strides[1] + 1
        channels = self.number_of_filters
        self.output_shape_value: tuple = (height, width, channels)
        return self.output_shape_value

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'
        return f"{self.name} (Conv2D){' ' * (28 - len(self.name) - 8)}{formatted_output}{' ' * (26-len(formatted_output))}{self.weights.size + self.biases.size}\n"

    def im2col_indices(self, inputs_shape: tuple) -> tuple[int]:
        height = (inputs_shape[0] - self.kernel_size[0]) // self.strides[0] + 1
        width = (inputs_shape[1] - self.kernel_size[1]) // self.strides[1] + 1
        channels = inputs_shape[-1]

        i0 = np.repeat(np.arange(self.kernel_size[0]), self.kernel_size[1])
        i0 = np.tile(i0, channels)
        i1 = self.strides[0] * np.repeat(np.arange(height), width)
        j0 = np.tile(
            np.arange(self.kernel_size[1]), self.kernel_size[0] * channels)
        j1 = self.strides[1] * np.tile(np.arange(width), height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(channels),
                      self.kernel_size[0] * self.kernel_size[1]).reshape(-1, 1)

        return (i, j, k)

    def im2col(self, x: np.ndarray) -> np.ndarray:
        """Support function to perform im2col operation for input data in Conv2D feed forward\n
        We use this as resarch proved it's faster to use im2col and then a simple matrix multiplication
        then sliding the kernel over the data and applying the filters manually

        Args:
            x (np.ndarray): Img we should perform the operation on. It should be 3d: height, width, channels

        Returns:
            np.ndarray: Columns calculated by the algorithm
        """
        i, j, k = self.im2col_indices(x.shape)
        return x[i, j, k]

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        input_shape = x.shape

        if is_training:
            self.inputs[self.current_batch] = x

        x_col = self.im2col(x)

        height = (input_shape[0] -
                  self.kernel_size[0]) // self.strides[0] + 1
        width = (input_shape[1] -
                 self.kernel_size[1]) // self.strides[1] + 1

        self.weights_col = self.weights.reshape(self.number_of_filters, -1)

        weighted_sum = np.dot(self.weights_col, x_col)

        weighted_sum = weighted_sum.reshape(
            self.number_of_filters, height, width).transpose(1, 2, 0)

        output = self.activation.apply_activation(weighted_sum)

        if is_training:
            self.outputs[self.current_batch] = output
            self.x_col[self.current_batch] = x_col
            self.current_batch += 1

        return output

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagate algorithm used for Conv2D layer.

        Args:
            gradient (np.ndarray): Gradient calculated by loss.compute_derivative() or previous layers output gradient
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient
        """
        inputs = np.average(self.inputs, axis=0)
        outputs = np.average(self.outputs, axis=0)
        x_col = np.average(np.array(self.x_col), axis=0)

        if self.regulizer:
            gradient = self.regulizer.update_gradient(
                gradient, self.weights, self.biases)

        delta = (gradient * self.activation.compute_derivative(outputs))

        weights_gradients = (delta.reshape(
            self.number_of_filters, -1) @ x_col.T).reshape(self.weights.shape)

        if self.trainable:
            self.weights, self.biases = optimizer[1].apply_gradients(
                weights_gradients, np.average(delta, (0, 1)), self.weights, self.biases)

        self.current_batch = 0

        return np.dot(np.average(delta), inputs)
