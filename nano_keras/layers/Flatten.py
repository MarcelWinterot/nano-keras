import numpy as np
from nano_keras.layers import Layer
from nano_keras.optimizers import Optimizer


class Flatten(Layer):
    """Flatten layer class. It's used to flatten the input of the model. It's input shape is (None, *input_shape) and it's output shape is (None, np.prod(input_shape))
    """

    def __init__(self, name: str = "Flatten") -> None:
        """Initalizer for the flatten layer

        Args:
            name (str, optional): Name of the layer. Defaults to "Flatten".
        """
        self.name: str = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        self.output_shape_value: int = np.prod(np.array(input_shape))
        self.next_layer_shape: tuple = layers[current_layer_index +
                                              1].output_shape(layers, current_layer_index+1)
        return self.output_shape_value

    def __repr__(self) -> str:
        return f"{self.name} (Flatten){' ' * (28 - len(self.name) - 9)}{(None, self.output_shape_value)}{' ' * (26-len(f'(None, {self.output_shape_value})'))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.original_shape: tuple = x.shape
        return np.ravel(x)

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagate algorithm for the flatten layer. We unflatten the gradient in here

        Args:
            gradient (np.ndarray): Gradient calculated by loss.compute_derivative() or previous layers output gradient
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient
        """
        try:
            return gradient.reshape(self.next_layer_shape, *self.original_shape)
        except ValueError:
            try:
                return gradient.reshape(*self.original_shape)
            except ValueError:
                return gradient
