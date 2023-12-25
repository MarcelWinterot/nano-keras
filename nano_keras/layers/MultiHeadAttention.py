import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.activations import Activation, ACTIVATIONS
from nano_keras.optimizers import Optimizer
from nano_keras.initializers import INITIALIZERS, Initializer


class MultiHeadAttention(LayerWithParams):
    def __init__(self, num_heads: int, key_dim: int, attention_axes: int | tuple = None, weight_initalizer: Initializer | str = "random_normal", bias_initalizer: Initializer | str = "zeros", name: str = "MultiHeadAttention") -> None:
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_axes = attention_axes

        self.weight_initalizer = INITIALIZERS[weight_initalizer] if type(
            weight_initalizer) == str else weight_initalizer
        self.bias_initalizer = INITIALIZERS[bias_initalizer] if type(
            bias_initalizer) == str else bias_initalizer

        self.name = name

        self.weights = np.array([])
        self.biases = np.array([])

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        self.output_shape_value = tuple(layers[current_layer_index -
                                               1].output_shape(layers, current_layer_index-1))

        return self.output_shape_value

    def __repr__(self) -> str:
        params_number = sum([
            weight.size for weight in self.weights]) + self.biases.size

        formatted_output = f"(None, {self.output_shape_value})"
        if type(self.output_shape_value) == tuple:
            formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'

        return f"{self.name} (MHA){' ' * (28 - len(self.name) - 5)}{formatted_output}{' ' * (26 - len(formatted_output))}{params_number}\n"

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_, bias_data_type: np.float_) -> None:
        input_shape = tuple(layers[current_layer_index -
                                   1].output_shape(layers, current_layer_index-1))

        self.weights = [np.random.randn(input_shape[-1], self.num_heads, self.key_dim).astype(weight_data_type) for _ in range(3)] + \
            [np.random.randn(self.num_heads, self.key_dim).astype(weight_data_type) for _ in range(3)] + \
            [np.random.randn(self.num_heads, self.key_dim,
                             input_shape[-1]).astype(weight_data_type)]

        self.biases = np.random.randn(
            input_shape[-1]).astype(bias_data_type)

        self.output_shape_value = tuple(layers[current_layer_index -
                                               1].output_shape(layers, current_layer_index-1))

    def attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        matmul_qk = np.matmul(q, k)

        dk = np.array(k.shape[-1])
        scaled_attention_logits = matmul_qk / np.sqrt(dk)

        attention_weights = ACTIVATIONS["softmax"].apply_activation(
            scaled_attention_logits)

        attended_values = np.matmul(attention_weights, v)

        return attended_values

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        raise NotImplementedError("MultiHeadAttention is not implemented yet")

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        raise NotImplementedError("MultiHeadAttention is not implemented yet")
