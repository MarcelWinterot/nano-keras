import numpy as np
from nano_keras.activations import Activation, ACTIVATIONS
from nano_keras.initializers import Initializer, INITIALIZERS
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer

"""Useful materials:
https://en.wikipedia.org/wiki/Gated_recurrent_unit
https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
"""


class GRU(LayerWithParams):
    def __init__(self, units: int, activation: Activation | str = "tanh", recurrent_actvation: Activation | str = "sigmoid", weight_initialization: Initializer | str = "random_normal", recurrent_weight_initialization: Initializer | str = "random_normal", bias_initalization: Initializer | str = "zeros", return_sequences: bool = True, regulizer: Regularizer = None, name: str = "GRU") -> None:
        self.units = units
        self.activation = activation if type(
            activation) == Activation else ACTIVATIONS[activation]
        self.recurrent_activation = recurrent_actvation if type(
            recurrent_actvation) == Activation else ACTIVATIONS[recurrent_actvation]

        self.weight_initialization = weight_initialization if type(
            weight_initialization) == Initializer else INITIALIZERS[weight_initialization]
        self.recurrent_weight_initialization = recurrent_weight_initialization if type(
            recurrent_weight_initialization) == Initializer else INITIALIZERS[recurrent_weight_initialization]
        self.bias_initialization: Initializer = bias_initalization if type(
            bias_initalization) == Initializer else INITIALIZERS[bias_initalization]

        self.return_sequences = return_sequences
        self.regulizer = regulizer
        self.name = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)

        self.output_shape_value = (
            input_shape[0], self.units) if self.return_sequences else self.units

        return self.output_shape_value

    def __repr__(self) -> str:
        formatted_output = f"(None, {self.output_shape_value})"
        if type(self.output_shape_value) == tuple:
            formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'

        return f"{self.name} (GRU){' ' * (28 - len(self.name) - 5)}{formatted_output}{' ' * (26 - len(formatted_output))}{self.input_weights.size + self.recurrent_weights.size + self.biases.size}\n"

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_, bias_data_type: np.float_) -> None:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)

        input_weights_shape = (input_shape[1], self.units)
        recurrent_weights_shape = (self.units, self.units)

        self.input_weights = np.random.randn(
            3, *input_weights_shape).astype(weight_data_type)
        self.recurrent_weights = np.random.randn(
            3, *recurrent_weights_shape).astype(weight_data_type)

        self.biases = np.random.randn(2, 3, self.units)

        self.hidden_state = np.zeros((input_shape[0], self.units))

        self.output_shape_value = (
            input_shape[0], self.units) if self.return_sequences else self.units

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        raise NotImplementedError(
            "GRU layer is not yet finished, please be patient")

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        raise NotImplementedError(
            "GRU layer is not yet finished, please be patient")
