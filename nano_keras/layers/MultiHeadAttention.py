import numpy as np
from nano_keras.layers import Layer, LayerWithParams, Dense
from nano_keras.activations import ACTIVATIONS
from nano_keras.optimizers import Optimizer
from nano_keras.initializers import INITIALIZERS, Initializer


class MultiHeadAttention(LayerWithParams):
    def __init__(self, num_heads: int, key_dim: int, value_dim: int = None, weight_initialization: Initializer | str = "random_normal", bias_initialization: Initializer | str = "zeros", name: str = "MultiHeadAttention") -> None:
        self.num_heads: int = num_heads
        self.key_dim: int = key_dim
        self.value_dim: int = value_dim if value_dim else key_dim

        self.weight_initialization: Initializer = INITIALIZERS[weight_initialization] if type(
            weight_initialization) == str else weight_initialization
        self.bias_initialization: Initializer = INITIALIZERS[bias_initialization] if type(
            bias_initialization) == str else bias_initialization

        self.name: str = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        self.output_shape_value = tuple(layers[current_layer_index -
                                               1].output_shape(layers, current_layer_index-1))

        return self.output_shape_value

    def __repr__(self) -> str:
        params_number = self.get_number_of_params()

        formatted_output = f"(None, {self.output_shape_value})"
        if type(self.output_shape_value) == tuple:
            formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'

        return f"{self.name} (MHA){' ' * (28 - len(self.name) - 5)}{formatted_output}{' ' * (26 - len(formatted_output))}{params_number}\n"

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_, bias_data_type: np.float_) -> None:
        input_shape = tuple(layers[current_layer_index -
                                   1].output_shape(layers, current_layer_index-1))

        self.query_weights = np.random.randn(
            input_shape[-1], self.num_heads, self.key_dim).astype(weight_data_type)
        self.query_biases = np.random.randn(
            self.num_heads, self.key_dim).astype(bias_data_type)

        self.key_weights = np.random.randn(
            input_shape[-1], self.num_heads, self.key_dim).astype(weight_data_type)
        self.key_biases = np.random.randn(
            self.num_heads, self.key_dim).astype(bias_data_type)

        self.value_weights = np.random.randn(
            input_shape[-1], self.num_heads, self.value_dim).astype(weight_data_type)
        self.value_biases = np.random.randn(
            self.num_heads, self.value_dim).astype(bias_data_type)

        self.output_weights = np.random.randn(
            self.num_heads, self.value_dim, input_shape[-1]).astype(weight_data_type)
        self.output_biases = np.random.randn(
            input_shape[-1]).astype(bias_data_type)

        self.output_shape_value = tuple(layers[current_layer_index -
                                               1].output_shape(layers, current_layer_index-1))

    def get_number_of_params(self) -> int:
        params_number = self.query_weights.size + self.query_biases.size + \
            self.key_weights.size + self.key_biases.size + \
            self.value_weights.size + self.value_biases.size + \
            self.output_weights.size + self.output_biases.size

        return params_number

    def get_params_size(self) -> int:
        params_size = self.query_weights.nbytes + self.query_biases.nbytes + \
            self.key_weights.nbytes + self.key_biases.nbytes + \
            self.value_weights.nbytes + self.value_biases.nbytes + \
            self.output_weights.nbytes + self.output_biases.nbytes

        return params_size

    def get_weights(self) -> list[np.ndarray]:
        return [self.query_weights, self.query_biases, self.key_weights, self.key_biases, self.value_weights, self.value_biases, self.output_weights, self.output_biases]

    def set_weights(self, query_weights: np.ndarray, query_biases: np.ndarray, key_weights: np.ndarray, key_biases: np.ndarray, value_weights: np.ndarray, value_biases: np.ndarray, output_weights: np.ndarray, output_biases: np.ndarray) -> None:
        self.query_weights = query_weights
        self.query_biases = query_biases
        self.key_weights = key_weights
        self.key_biases = key_biases
        self.value_weights = value_weights
        self.value_biases = value_biases
        self.output_weights = output_weights
        self.output_biases = output_biases

    def compute_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        matmul_qk = np.matmul(q, k.transpose(0, 2, 1))

        dk = np.array(k.shape[-1])
        scaled_attention_logits = matmul_qk / np.sqrt(dk)

        attention_weights = ACTIVATIONS["softmax"].apply_activation(
            scaled_attention_logits)

        attended_values = np.matmul(attention_weights, v)

        return attended_values

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.inputs = x

        Q = np.dot(x, self.query_weights.transpose(1, 0, 2)) + \
            self.query_biases
        K = np.dot(x, self.key_weights.transpose(1, 0, 2)) + self.key_biases
        V = np.dot(x, self.value_weights.transpose(1, 0, 2)) + \
            self.value_biases

        Q = np.array(np.split(Q.transpose(0, 2, 1),
                     self.num_heads, axis=-1))[:, :, :, 0]
        K = np.array(np.split(K.transpose(0, 2, 1),
                     self.num_heads, axis=-1))[:, :, :, 0]
        V = np.array(np.split(V.transpose(0, 2, 1),
                     self.num_heads, axis=-1))[:, :, :, 0]

        attention = self.compute_attention(Q, K, V)

        self.attention_output = np.concatenate(attention, axis=-1)

        output_weights = self.output_weights.transpose(1, 0, 2)

        self.output = np.dot(self.attention_output, output_weights.reshape(-1,
                                                                           output_weights.shape[-1])) + self.output_biases

        return self.output

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        return gradient
