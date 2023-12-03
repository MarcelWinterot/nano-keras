import numpy as np
from nano_keras.layers import Layer
from nano_keras.activations import Activation
from nano_keras.regulizers import Regularizer


class LayerWithParams(Layer):
    def __init__(self, units: int, activation: Activation | str, weight_initialization: str = "random", regulizer: Regularizer = None, name: str = "Dense") -> None:
        super().__init__(units, activation, weight_initialization, regulizer, name)
        self.weights = np.array([])
        self.biases = np.array([])
