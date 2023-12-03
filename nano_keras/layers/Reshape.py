import numpy as np
from nano_keras.layers import Layer
from nano_keras.optimizers import Optimizer


class Reshape(Layer):
    def __init__(self, target_shape: tuple, name: str = "Reshape") -> None:
        """Initalizer for the Reshape layer

        Args:
            target_shape (tuple): Target shape of the layer
            name (str, optional): Name of the layer. Defaults to "Reshape".
        """
        self.target_shape: tuple = target_shape
        self.name: str = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        self.next_layer_shape: tuple = layers[current_layer_index +
                                              1].output_shape(layers, current_layer_index+1)
        return self.target_shape

    def __repr__(self) -> str:
        formatted_output = f'(None, {", ".join(map(str, self.target_shape))})'
        return f"{self.name} (Reshape){' ' * (28 - len(self.name) - 9)}{formatted_output}{' ' * (26-len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.original_shape: tuple = x.shape
        return np.reshape(x, self.target_shape)

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagate algorithm used for reshape layer. We reshape the gradient in here

        Args:
            gradient (np.ndarray): Gradient calculated by the loss.compute_derivative function
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient
        """
        try:
            return gradient.reshape(self.next_layer_shape, *self.original_shape)
        except:
            return gradient
