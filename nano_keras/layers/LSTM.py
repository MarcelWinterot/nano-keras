import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.activations import Activation, ACTIVATIONS
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer


class LSTM(LayerWithParams):
    def __init__(self, units: int, activation: Activation | str = "sigmoid", recurrent_activation: Activation | str = "tanh", weight_initalization: str = "random", recurrent_weight_initalization: str = "random", return_sequences: bool = True, regulizer: Regularizer = None, name: str = "LSTM") -> None:
        self.units = units
        self.activation = activation if type(
            activation) == Activation else ACTIVATIONS[activation]
        self.recurrent_activation = recurrent_activation if type(
            recurrent_activation) == Activation else ACTIVATIONS[recurrent_activation]
        self.weight_initialization = weight_initalization
        self.recurrent_weight_initalization = recurrent_weight_initalization
        self.return_sequences = return_sequences
        self.regulizer = Regularizer
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

        return f"{self.name} (LSTM){' ' * (28 - len(self.name) - 6)}{formatted_output}{' ' * (26 - len(formatted_output))}{self.input_weights.size + self.recurrent_weights.size + self.biases.size}\n"

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_) -> None:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)

        input_weights_shape = (self.units, self.units)
        recurrent_weights_shape = (input_shape[1], self.units)

        self.input_weights = np.random.randn(
            4, *input_weights_shape).astype(weight_data_type)
        self.recurrent_weights = np.random.randn(
            4, *recurrent_weights_shape).astype(weight_data_type)

        self.biases = np.random.randn(4, 1, self.units)

        return

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        return super().__call__(x, is_training)
