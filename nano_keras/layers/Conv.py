import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer
from nano_keras.activations import Activation, ACTIVATIONS


class Conv1D(LayerWithParams):
    def __init__(self, filters: int = 1, kernel_size: int = 2, strides: int = 2, activation: Activation | str = "relu", weight_initialization: str = "he", regulizer: Regularizer = None, name: str = "Conv1D") -> None:
        """Initalizer for the Conv1D layer

        Args:
            filters (int, optional): Number of filters. Defaults to 1.
            kernel_size (int, optional): Kernel size the model should use. Defaults to 2.
            strides (int, optional): By how much should the kernel move after each operation. Defaults to 2.
            activation (Activation | str, optional): Activation function the layer should use. Defaults to "relu".
            weight_initaliziton (str, optional): Weights intialization strategy you want to use to generate weights of the layer. Your options are: random, xavier, he. Defalut to "he"
            regulizer (Regularizer, optional): Regulizer of the layer. Defaults to None.
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
        self.name: str = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
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

        self.weights, self.biases = optimizer[0].apply_gradients(weights_gradients, np.array(
            delta, dtype=float), self.weights, self.biases)

        return np.dot(delta, self.weights.T)


class Conv2D(LayerWithParams):
    def __init__(self, filters: int = 1, kernel_size: tuple = (2, 2), strides: tuple = (1, 1), activation: Activation | str = "relu", weight_initaliziton: str = "he", regulizer: Regularizer = None, name: str = "Conv2D") -> None:
        """Intializer for the Conv2D layer

        Args:
            filters (int, optional): Amount of filters in the layer. Defaults to 1.
            kernel_size (tuple, optional): Kernel size the layer should use. Defaults to (2, 2).
            strides (tuple, optional): By how much should the kernel move. Defaults to (1, 1).
            activation (Activation | str, optional): Activation function of the layer. Defaults to "relu".
            weight_initaliziton (str, optional): Weights intialization strategy you want to use to generate weights of the layer. Your options are: random, xavier, he. Defalut to "he"
            regulizer (Regularizer, optional): Regulizer for the layer. Defaults to None.
            name (str, optional): Name of the layer. Defaults to "Conv2D".
        """
        self.number_of_filters: int = filters
        self.kernel_size: tuple = kernel_size
        self.strides: tuple = strides
        self.activation: Activation = ACTIVATIONS[activation] if type(
            activation) == str else activation
        self.weight_initialization = weight_initaliziton
        self.regulizer: Regularizer = regulizer
        self.name: str = name
        self.weights: np.ndarray = np.array([])
        self.biases: np.ndarray = np.array([])

    def random_initalization(self, weights: list, input_shape: tuple, weight_data_type: np.float_) -> tuple[np.ndarray, np.ndarray]:
        """Random intitalization strategy used for weights generation. Note that this works for layers that don't use units for weight generation like Conv1d and Conv2d

        Args:
            weights (list): Weights shape of the layer
            input_shape (tuple): Input shape of the layer, it's here as it's in the 2 other intialization startegies and it saves as an if
            weight_data_type (np.float_): Data type of the weights. Remember to use np.float32 or np.float64

        Returns:
            tuple[np.ndarray, np.ndarray]: Weights of the layer
        """
        return np.random.randn(*weights).astype(weight_data_type), np.random.randn(self.number_of_filters).astype(weight_data_type)

    def xavier_intialization(self, weights: list, input_shape: tuple, weight_data_type: np.float_) -> tuple[np.ndarray, np.ndarray]:
        """Xavier intitalization strategy used for weights generation. Note that this works for layers that don't use units for weight generation like Conv1d and Conv2d

        Args:
            weights (list): Weights shape of the layer
            input_shape (tuple): Input shape of the layer, which we use to calculate fan_in
            weight_data_type (np.float_): Data type of the weights. Remember to use np.float32 or np.float64

        Returns:
            tuple[np.ndarray, np.ndarray]: Weights of the layer
        """
        weights = np.random.randn(*weights).astype(weight_data_type)
        weights = 2 * weights - 1
        fan_in = input_shape[-1] * self.kernel_size[0] * self.kernel_size[1]
        weights *= np.sqrt(6/(fan_in + self.number_of_filters))
        return weights, np.zeros(self.number_of_filters).astype(weight_data_type)

    def he_intialization(self, weights: list, input_shape: tuple, weight_data_type: np.float_) -> tuple[np.ndarray, np.ndarray]:
        """He intitalization strategy used for weights generation. Note that this works for layers that don't use units for weight generation like Conv1d and Conv2d

        Args:
            weights (list): Weights shape of the layer
            input_shape (tuple): Input shape of the layer, which we use to calculate fan_in
            weight_data_type (np.float_): Data type of the weights. Remember to use np.float32 or np.float64

        Returns:
            tuple[np.ndarray, np.ndarray]: Weights of the layer
        """
        weights = np.random.randn(*weights).astype(weight_data_type)
        fan_in = input_shape[-1] * self.kernel_size[0] * self.kernel_size[1]
        weights *= np.sqrt(2. / fan_in)
        return weights, np.zeros(self.number_of_filters).astype(weight_data_type)

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_) -> None:
        LAYER_INTIALIZATIONS = {"random": self.random_initalization,
                                "xavier": self.xavier_intialization, "he": self.he_intialization}

        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)

        weights = (self.kernel_size[0], self.kernel_size[1],
                   input_shape[-1], self.number_of_filters)

        self.weights, self.biases = LAYER_INTIALIZATIONS[self.weight_initialization](
            weights, input_shape, weight_data_type)

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        self.input_shape: tuple = layers[current_layer_index -
                                         1].output_shape(layers, current_layer_index-1)
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

    def im2col(self, x: np.ndarray) -> np.ndarray:
        """Support function to perform im2col operation for input data in Conv2D feed forward\n
        We use this as resarch proved it's faster to use im2col and then a simple matrix multiplication
        then sliding the kernel over the data and applying the filters manually

        Args:
            x (np.ndarray): Img we should perform the operation on. It should be 3d: height, width, channels

        Returns:
            np.ndarray: Columns calculated by the algorithm
        """
        inputs_shape = x.shape
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

        return x[i, j, k]

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Call function also known as feed forward function for the Conv2D layer

        Args:
            x (np.ndarray): X dataset

        Returns:
            np.ndarray: Layers output
        """
        input_shape = x.shape

        self.inputs: np.ndarray = x

        self.x_col = self.im2col(x)

        height = (input_shape[0] -
                  self.kernel_size[0]) // self.strides[0] + 1
        width = (input_shape[1] -
                 self.kernel_size[1]) // self.strides[1] + 1

        weights_col = self.weights.reshape(self.number_of_filters, -1)

        weighted_sum = np.dot(weights_col, self.x_col)

        weighted_sum = weighted_sum.reshape(
            self.number_of_filters, height, width).transpose(1, 2, 0)

        self.output = self.activation.apply_activation(weighted_sum)

        return self.output

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagate algorithm used for Conv2D layer. It's kinda slow right now so it's not recommended to use it but it will be faster in the future

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

        delta = (gradient * self.activation.compute_derivative(self.output)
                 ).reshape(self.number_of_filters, -1)

        weights_gradients = (delta @ self.x_col.T).reshape(self.weights.shape)

        averaged_delta = np.average(delta)

        self.weights, self.biases = optimizer[1].apply_gradients(
            weights_gradients, averaged_delta, self.weights, self.biases)

        return np.dot(averaged_delta, self.inputs)
