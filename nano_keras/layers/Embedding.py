import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer
from nano_keras.initializers import Initializer, INITIALIZERS


class Embedding(LayerWithParams):
    """Embedding layer. It's a lookup table that maps integers to vectors. It's input shape is (None, input_length) and it's output shape is (None, input_length, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, embedding_initalizer: Initializer | str = "random_normal", regulizer: Regularizer = None, input_length: int = None, trainable: bool = True, name: str = "Embedding") -> None:
        """Initalizer for the Embedding layer

        Args:
            input_dim (int): Size of the vocabulary, i.e. maximum integer index + 1.
            output_dim (int): Dimension of the dense embedding.
            embedding_initalizer (Initializer | str, optional): Initalizer for the lookup table/weights. Defaults to "random_normal".
            regulizer (Regularizer, optional): Regulizer the model should use. You can find them all in the regulizers.py file. You must pass the already intialized class. Defaults to None.
            input_length (int, optional): Length of the input. Defaults to None.
            trainable (bool, optional): Parameter that decides whether the parameters should be updated or no. Defaults to True.
            name (str, optional): Name of the layer. Helpful for debugging. Defaults to "Embedding".
        """
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.input_length: int = input_length

        self.embedding_initalizer: Initializer = embedding_initalizer if type(
            embedding_initalizer) == Initializer else INITIALIZERS[embedding_initalizer]

        self.regulizer: Regularizer = regulizer
        self.trainable: bool = trainable
        self.name: str = name

        self.weights: np.ndarray = np.array([])
        # Added biases as I don't want to add a new class for this
        self.biases: np.ndarray = np.array([])

        self.batch_size: int = 1
        self.current_batch: int = 0

    def set_batch_size(self, batch_size: int, layers: list, index: int) -> None:
        self.batch_size = batch_size

        if self.input_length:
            input_shape = self.input_length
        else:
            input_shape = layers[index - 1].output_shape(layers, index-1)

        self.inputs = np.ndarray((self.batch_size, *input_shape)) if type(
            input_shape) == tuple else np.ndarray((self.batch_size, input_shape))

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        if self.input_length:
            input_shape = self.input_length
        else:
            input_shape = layers[current_layer_index -
                                 1].output_shape(layers, current_layer_index-1)

        self.output_shape_value = (input_shape, self.output_dim)

        return self.output_shape_value

    def __repr__(self) -> str:
        formatted_output = f"(None, {self.output_shape_value})"
        if type(self.output_shape_value) == tuple:
            formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'

        return f"{self.name} (Embedding){' ' * (28 - len(self.name) - 11)}{formatted_output}{' ' * (26 - len(formatted_output))}{self.weights.size}\n"

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_, bias_data_type: np.float_) -> None:
        if self.input_length:
            input_shape = self.input_length
        else:
            input_shape = layers[current_layer_index -
                                 1].output_shape(layers, current_layer_index-1)

        self.weights = self.embedding_initalizer(
            (self.input_dim, self.output_dim), weight_data_type)

        self.output_shape_value = (input_shape, self.output_dim)

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.inputs[self.current_batch] = x

        return self.weights[x]

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        """Backpropagation algorithm for the Embedding layer

        Args:
            gradient (np.ndarray): gradient calculated by losses.compute_derivative()
            optimizer (Optimizer): Optimizer to use when updating layers parameters

        Returns:
            np.ndarray: New gradient
        """
        if self.regulizer:
            self.regulizer.regulize(self.weights)

        inputs = (np.sum(self.inputs, axis=0) // self.batch_size).astype(int)

        weights_gradients = np.zeros_like(self.weights)
        np.add.at(weights_gradients, inputs, gradient)

        if self.trainable:
            self.weights, _ = optimizer[0].apply_gradients(
                weights_gradients, [], self.weights, [], False)

        self.current_batch = 0

        return gradient
