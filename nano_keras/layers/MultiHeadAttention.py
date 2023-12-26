import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.activations import ACTIVATIONS
from nano_keras.optimizers import Optimizer
from nano_keras.initializers import INITIALIZERS, Initializer


class MultiHeadAttention(LayerWithParams):
    def __init__(self, num_heads: int, key_dim: int, weight_initalizer: Initializer | str = "random_normal", bias_initalizer: Initializer | str = "zeros", name: str = "MultiHeadAttention") -> None:
        self.num_heads: int = num_heads
        self.key_dim: int = key_dim

        self.weight_initalizer: Initializer = INITIALIZERS[weight_initalizer] if type(
            weight_initalizer) == str else weight_initalizer
        self.bias_initalizer: Initializer = INITIALIZERS[bias_initalizer] if type(
            bias_initalizer) == str else bias_initalizer

        self.name: str = name

        self.weights: np.ndarray = np.array([])
        self.biases: np.ndarray = np.array([])

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

        self.weights = self.weight_initalizer(
            (3, *input_shape), weight_data_type)

        self.biases = self.bias_initalizer(input_shape[-1], bias_data_type)

        self.output_shape_value = tuple(layers[current_layer_index -
                                               1].output_shape(layers, current_layer_index-1))

    def attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, *args, **kwargs) -> np.ndarray:
        matmul_qk = np.matmul(q, k.T)

        dk = np.array(k.shape[-1])
        scaled_attention_logits = matmul_qk / np.sqrt(dk)

        attention_weights = ACTIVATIONS["softmax"].apply_activation(
            scaled_attention_logits)

        attended_values = np.matmul(attention_weights, v)

        try:
            index = kwargs["index"]
            attended_values += self.biases_split[index]
        except KeyError:
            pass

        return attended_values

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        Q, K, V = np.copy(x), np.copy(x), np.copy(x)

        Q = Q * self.weights[0]
        K = K * self.weights[1]
        V = V * self.weights[2]

        Q_split = np.array_split(Q, self.num_heads, axis=-1)
        K_split = np.array_split(K, self.num_heads, axis=-1)
        V_split = np.array_split(V, self.num_heads, axis=-1)

        self.biases_split = np.array_split(self.biases, self.num_heads)

        output = [self.attention(q, k, v, index=i) for i, (q, k, v) in enumerate(
            zip(Q_split, K_split, V_split))]

        output = np.concatenate(output, axis=-1)

        return output
