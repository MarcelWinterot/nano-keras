import numpy as np
from nano_keras.layers import Layer
from nano_keras.activations import Activation
from nano_keras.regulizers import Regularizer
from nano_keras.initializers import Initializer


class LayerWithParams(Layer):
    """Base class used to build new layers with parameters.
    It's used with all the layers that have parameters to update
    """

    def __init__(self, units: int, activation: Activation | str, weight_initialization: Initializer | str = "random_normal", bias_initalization: Initializer | str = "random_normal", regulizer: Regularizer = None, trainable: bool = True, input_shape: tuple = None, name: str = "Layer") -> None:
        super().__init__(units, activation, weight_initialization,
                         bias_initalization, regulizer, trainable, input_shape, name)
        self.weights = np.array([])
        self.biases = np.array([])

    def get_number_of_params(self) -> tuple:
        if self.trainable:
            return (self.weights.size + self.biases.size, 0)
        return (0, self.weights.size + self.biases.size)

    def get_params_size(self) -> tuple:
        if self.trainable:
            return (self.weights.nbytes + self.biases.nbytes, 0)
        return (0, self.weights.nbytes + self.biases.nbytes)

    def get_weights(self) -> list[np.ndarray]:
        return [self.weights, self.biases]

    def set_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        """Function used to set the weights and biases of the layer

        Args:
            weights (np.ndarray): Weights of the layer. If their shape is different than the current weights shape, the feedforward will fail
            biases (np.ndarray): Biases of the layer. If their shape is different than the current biases shape, the feedforward will fail
        """
        self.weights = weights
        self.biases = biases
