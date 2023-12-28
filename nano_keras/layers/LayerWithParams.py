import numpy as np
from nano_keras.layers import Layer
from nano_keras.activations import Activation
from nano_keras.regulizers import Regularizer
from nano_keras.initializers import Initializer


class LayerWithParams(Layer):
    def __init__(self, units: int, activation: Activation | str, weight_initialization: Initializer | str = "random_normal", bias_initalization: Initializer | str = "random_normal", regulizer: Regularizer = None, trainable: bool = True, name: str = "Dense") -> None:
        super().__init__(units, activation, weight_initialization,
                         bias_initalization, regulizer, trainable, name)
        self.weights = np.array([])
        self.biases = np.array([])

    def get_number_of_params(self) -> int:
        return self.weights.size + self.biases.size

    def get_params_size(self) -> int:
        return self.weights.nbytes + self.biases.nbytes

    def get_weights(self) -> list[np.ndarray]:
        return [self.weights, self.biases]

    def set_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases
