import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer
from nano_keras.initializers import Initializer, INITIALIZERS


class Embedding(LayerWithParams):
    def __init__(self, input_dim: int, output_dim: int, embedding_initalizer: Initializer | str = "random_normal", regulizer: Regularizer = None, input_length: int = None, name: str = "Embedding") -> None:
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.input_length: int = input_length

        self.embedding_initalizer: Initializer = embedding_initalizer if type(
            embedding_initalizer) == Initializer else INITIALIZERS[embedding_initalizer]

        self.regulizer: Regularizer = regulizer
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
        if self.regulizer:
            self.regulizer.regulize(self.weights)

        inputs = (np.sum(self.inputs, axis=0) // self.batch_size).astype(int)

        weights_gradients = np.zeros_like(self.weights)
        np.add.at(weights_gradients, inputs, gradient)

        self.weights, _ = optimizer[0].apply_gradients(
            weights_gradients, [], self.weights, [], False)

        self.current_batch = 0

        return gradient
