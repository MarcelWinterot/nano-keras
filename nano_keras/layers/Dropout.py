import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.activations import Activation
from nano_keras.regulizers import Regularizer
from nano_keras.optimizers import Optimizer


class Dropout(LayerWithParams):
    def __init__(self, units: int, activation: Activation | str, dropout_rate: float = 0.2, weight_initialization: str = "random", regulizer: Regularizer | None = None, name: str = "Dropout") -> None:
        """Intializer for the dropout layer. Note that dropout layer acts the same as Dense but also drops connections between neurons

        Args:
            units (int): Number of neurons in the layer
            activation (Activation | str): ACtivation function the model should use
            dropout_rate (float, optional): The pecentage of connections dropped. Note that the number after the dot is the percentage. Defaults to 0.2.
            weight_initaliziton (str, optional): Weights intialization strategy you want to use to generate weights of the layer. Your options are: random, xavier, he. Defalut to "random"
            regulizer (Regularizer | None, optional): Regulizer the model should use. Defaults to None.
            name (str, optional): Name of the layer. Defaults to "Layer".
        """
        super().__init__(units, activation, weight_initialization, regulizer, name)
        self.dropout_rate: float = dropout_rate

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        return self.units

    def __repr__(self) -> str:
        return f"{self.name} (Dropout){' ' * (28 - len(self.name) - 9)}{(None, self.units)}{' ' * (26 - len(f'(None, {self.units})'))}{self.weights.size + self.biases.size}\n"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Call function for the dropout layer. Dropout feed forward works simmilarly to Dense layers feed forward, with only difference being is that we apply a mask to drop connections\n

        Args:
            x (np.ndarray): X dataset
            is_training (bool, optional): Param to control the mask. If it's set to False we don't apply the mask. Defaults to False.

        Returns:
            np.ndarray: Output of the model
        """
        self.inputs: np.ndarray = x

        weighted_sum = np.dot(x, self.weights) + self.biases

        if is_training:
            weighted_sum *= (np.random.rand(*weighted_sum.shape)
                             >= self.dropout_rate).astype(np.float64)

            weighted_sum /= (1 - self.dropout_rate)

        self.output = self.activation.apply_activation(weighted_sum)
        self.is_training = is_training

        return self.output

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagation algorithm for the dropout layer

        Args:
            gradient (np.ndarray): Gradient calculated by loss.compute_derivative() or previous layers output gradient
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient
        """
        if self.regulizer:
            gradient = self.regulizer.update_gradient(
                gradient, self.weights, self.biases)

        delta = gradient * self.activation.compute_derivative(self.output)

        if self.is_training:
            delta /= (1 - self.dropout_rate)

        weights_gradients = np.outer(self.inputs, delta)

        self.weights, self.biases = optimizer[0].apply_gradients(
            weights_gradients, np.average(delta), self.weights, self.biases)

        return np.dot(delta, self.weights.T)
