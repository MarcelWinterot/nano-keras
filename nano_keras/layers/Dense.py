from nano_keras.activations import Activation
from nano_keras.initializers import Initializer
from nano_keras.layers import Layer, LayerWithParams
import numpy as np
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer


class Dense(LayerWithParams):
    """Dense/Linear/Fully connected layer. The input shape is (None, input_shape) and the output shape is (None, units)
    """

    def __init__(self, units: int, activation: Activation | str, weight_initialization: Initializer | str = "random_normal", bias_initalization: Initializer | str = "random_normal", regulizer: Regularizer = None, trainable: bool = True, input_shape: tuple = None, name: str = "Dense") -> None:
        """Initalizer for the Dense class

        Args:
            units (int): Number of neurons the layer should have
            activation (Activation | str): Activation function the model should use. You can find them all in the activations.py.
            weight_initaliziton (str, optional): Weights intialization strategy you want to use to generate weights of the layer. You can find all of them in the Initalizers folder. Defalut to "random_normal"
            bias_initialization (Initalizer | str, optional): Weights intialization strategy you want to use to generate biases of the layer. You can find all of them in the Initalizers folder. Defalut to "random_normal"
            regulizer (Regularizer, optional): Regulizer the model should use. You can find them all in the regulizers.py file. You must pass the already intialized class. Defaults to None.
            trainable (bool, optional): Parameter that decides whether the parameters should be updated or no. Defaults to True.
            input_shape (tuple, optional): Input shape to the layer. Used if you dont't want to use Input layer. If it's None it won't be used. Defaults to None.
            name (str, optional): Name of the layer. Helpful for debugging. Defaults to "Dense".
        """
        super().__init__(units, activation, weight_initialization,
                         bias_initalization, regulizer, trainable, input_shape, name)

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        return self.units

    def __repr__(self) -> str:
        return f"{self.name} (Dense){' ' * (28 - len(self.name) - 7)}{(None, self.units)}{' ' * (26 - len(f'(None, {self.units})'))}{self.weights.size + self.biases.size}\n"

    def backpropagate(self, gradient: np.ndarray, optimizer: list[Optimizer]) -> np.ndarray:
        """Backpropagation algorithm for the dense layer

        Args:
            gradient (np.ndarray): Gradient calculated by the loss function derivative or by previous layers backpropagation algorithm
            optimizer (List[Optimizer]): Optimizer to use for updating the model's parameters. Note that we use 2 different optimizers as then we don't have to check a bunch of times 
            wheter we use 1 or 2 optimizers, and we need 2 optimizers for CNNs

        Returns:
            np.ndarray: Output gradient of the layer
        """
        inputs = np.average(self.inputs, axis=0)
        outputs = np.average(self.outputs, axis=0)

        if self.regulizer:
            gradient = self.regulizer.update_gradient(
                gradient, self.weights, self.biases)

        delta = gradient * self.activation.compute_derivative(outputs)

        weights_gradients = np.outer(inputs, delta)

        if self.trainable:
            self.weights, self.biases = optimizer[0].apply_gradients(
                weights_gradients, np.average(delta), self.weights, self.biases)
        else:
            # We have to still call the optimizer to update the moving averages
            # As otherwise we'd get an error with shapes of the moving averages of biases
            _, _ = optimizer[0].apply_gradients(
                weights_gradients, np.average(delta), np.copy(self.weights), np.copy(self.biases))

        self.current_batch = 0

        return np.dot(delta, self.weights.T)
