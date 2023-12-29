from nano_keras.layers import Layer
import numpy as np


class Input(Layer):
    """Input layer class. It's used to define the input shape of the model and pass the input to the model.
    """

    def __init__(self, input_shape: tuple | int, name: str = "Input") -> None:
        """Intializer for input layer.

        Args:
            input_shape (tuple | int): Shape of the input for the model. If you have a 1d shape just use int instead of tuple.
            name (str, optional): Name of the layer. Defaults to "Input".
        """
        self.input_shape: tuple = input_shape
        self.name: str = name
        self.biases: np.ndarray = np.random.randn(
            *input_shape) if type(input_shape) == tuple else np.random.randn(input_shape)

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        return self.input_shape

    def __repr__(self) -> str:
        try:
            formatted_output = f'(None, {", ".join(map(str, self.input_shape))})'
        except:
            formatted_output = f'(None, {self.input_shape})'
        return f"{self.name} (Input){' ' * (28 - len(self.name) - 7)}{formatted_output}{' ' * (26 - len(formatted_output))}0\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        return x
