import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer
from nano_keras.initializers import Initializer, INITIALIZERS


class Embedding(LayerWithParams):
    def __init__(self, input_dim: int, output_dim: int, embedding_initalizer: Initializer | str = "random_normal", regulizer: Regularizer = None, input_length: int = None, name: str = "Embedding") -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embedding_initalizer = embedding_initalizer if type(
            embedding_initalizer) == Initializer else INITIALIZERS[embedding_initalizer]

        self.regulizer = regulizer
        self.input_length = input_length
        self.name = name

        self.weights = np.array([])
        self.biases = np.array([])

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

        self.weights = np.random.randn(
            self.input_dim, self.output_dim).astype(weight_data_type)

        self.output_shape_value = (input_shape, self.output_dim)

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        if is_training:
            self.inputs = x

        self.output = self.weights[x]

        return self.output

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        raise NotImplementedError("Embedding layer is not finished yet")