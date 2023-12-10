import numpy as np
from nano_keras.layers import Layer
from nano_keras.activations import Activation
from nano_keras.regulizers import Regularizer
from nano_keras.initializers import Initializer


class LayerWithParams(Layer):
    def __init__(self, units: int, activation: Activation | str, weight_initialization: Initializer | str = "random_normal", bias_initalization: Initializer | str = "random_normal", regulizer: Regularizer = None, name: str = "Dense") -> None:
        super().__init__(units, activation, weight_initialization,
                         bias_initalization, regulizer, name)
        self.weights = np.array([])
        self.biases = np.array([])
