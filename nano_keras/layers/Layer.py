import numpy as np
from nano_keras.activations import Activation, ACTIVATIONS
from nano_keras.regulizers import Regularizer
from nano_keras.optimizers import Optimizer


class Layer:
    def __init__(self, units: int, activation: Activation | str, weight_initialization: str = "random", regulizer: Regularizer = None, name: str = "Dense") -> None:
        """Intializer for the layer class. 

        Args:
            units (int): Number of neurons the layer should have
            activation (Activation | str): Activation function the model should use. You can find them all in the activations.py.
            weight_initaliziton (str, optional): Weights intialization strategy you want to use to generate weights of the layer. Your options are: random, xavier, he. Defalut to "random"
            regulizer (Regularizer, optional): Regulizer the model should use. You can find them all in the regulizers.py file. You must pass the already intialized class. Defaults to None.
            name (str, optional): Name of the layer. Helpful for debugging. Defaults to "Dense".
        """
        self.units: int = units
        self.name: str = name
        self.weight_initialization: str = weight_initialization
        self.activation: Activation = ACTIVATIONS[activation] if type(
            activation) == str else activation
        self.regulizer: Regularizer = regulizer
        self.batch_size: int = 1
        # Going from 0 to batch_size as it represent the index of self.inputs and self.outputs
        self.current_batch: int = 0

    def set_batch_size(self, batch_size: int, layers: list, index: int) -> None:
        self.batch_size = batch_size

        input_shape = layers[index-1].output_shape(layers, index-1)
        output_shape = self.output_shape(layers, index)

        self.inputs = np.ndarray((self.batch_size, *input_shape)) if type(
            input_shape) == tuple else np.ndarray((self.batch_size, input_shape))
        self.outputs = np.ndarray((self.batch_size, *output_shape)) if type(
            output_shape) == tuple else np.ndarray((self.batch_size, output_shape))

    @staticmethod
    def random_initalization(previous_units: int, current_units: int, weight_data_type: np.float_) -> tuple[np.ndarray, np.ndarray]:
        """Random intitalization strategy used for weights generation. Note that this works for 2d layers that use units for weight generation and not layers like Conv1d and Conv2d

        Args:
            previous_units (int): Number of units in the previous layer
            current_units (int): Number of units in the current layer in which we want the weights generated
            weight_data_type (np.float_): In what data type do we want the weights stored

        Returns:
            tuple[np.ndarray, np.ndarray]: Weights and biases of the layer
        """
        return np.random.randn(previous_units, current_units).astype(weight_data_type), np.random.randn(current_units).astype(weight_data_type)

    @staticmethod
    def xavier_intialization(previous_units: int, current_units: int, weight_data_type: np.float_) -> tuple[np.ndarray, np.ndarray]:
        """Xavier intitalization strategy used for weights generation. Note that this works for 2d layers that use units for weight generation and not layers like Conv1d and Conv2d

        Args:
            previous_units (int): Number of units in the previous layer
            current_units (int): Number of units in the current layer in which we want the weights generated
            weight_data_type (np.float_): In what data type do we want the weights stored

        Returns:
            tuple[np.ndarray, np.ndarray]: Weights and biases of the layer
        """
        weights = np.random.randn(
            previous_units, current_units).astype(weight_data_type)
        weights = 2 * weights - 1
        weights *= np.sqrt(6/(previous_units+current_units))
        return weights, np.zeros(current_units).astype(weight_data_type)

    @staticmethod
    def he_intialization(previous_units: int, current_units: int, weight_data_type: np.float_) -> tuple[np.ndarray, np.ndarray]:
        """He intitalization strategy used for weights generation. Note that this works for 2d layers that use units for weight generation and not layers like Conv1d and Conv2d

        Args:
            previous_units (int): Number of units in the previous layer
            current_units (int): Number of units in the current layer in which we want the weights generated
            weight_data_type (np.float_): In what data type do we want the weights stored

        Returns:
            tuple[np.ndarray, np.ndarray]: Weights and biases of the layer
        """
        weights = np.random.randn(
            previous_units, current_units).astype(weight_data_type)
        weights *= np.sqrt(2./previous_units)
        return weights, np.zeros(current_units).astype(weight_data_type)

    def generate_weights(self, layers: list, current_layer_index: int, weight_data_type: np.float_) -> None:
        """Function used for weights generation for layers with 2d weights generated by looking at current layer and previous layers amount of neurons

        Args:
            layers (list): All layers in the model
            current_layer_index (int): For what layer do we want to generate the weights
            weight_data_type (np.float_): In what data type do you want to store the weights. Only use datatypes like np.float32 and np.float64
        """
        LAYER_INTIALIZATIONS = {"random": self.random_initalization,
                                "xavier": self.xavier_intialization, "he": self.he_intialization}

        previous_units = layers[current_layer_index -
                                1].output_shape(layers, current_layer_index-1)

        self.weights, self.biases = LAYER_INTIALIZATIONS[self.weight_initialization](
            previous_units, self.units, weight_data_type)

    def output_shape(self, layers: list, current_layer_index: int) -> tuple:
        """Function to generate the output shape of a layer

        Args:
            layers (list): All layers in the model
            current_layer_index (int): Index of the current layer

        Returns:
            tuple: Output shape of the layer
        """
        return

    def __repr__(self) -> str:
        """Function to print out layer information

        Returns:
            str: What to show when using print()
        """
        return "Base layer class"

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Call function for the layer, also know as feed forward\n
        Note that we also store all the variables the models calculated in self as it's layer used in backpropagate

        Args:
            x (np.ndarray): X dataset
            is_training (bool): Determines whether the layer should behave like in the training loop or no. Turning it off\n
            might give better results

        Returns:
            np.ndarray: output of the model
        """
        weighted_sum = np.dot(x, self.weights) + self.biases
        output = self.activation.apply_activation(
            weighted_sum)

        if is_training:
            self.inputs[self.current_batch] = x
            self.outputs[self.current_batch] = output
            self.current_batch += 1

        return output

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        """Backpropagation algorithm base implementation for all the layers that don't have any parameters to update

        Args:
            gradient (np.ndarray): gradient calculated by losses.compute_derivative()
            optimizer (Optimizer): Optimizer to use when updating layers parameters

        Returns:
            np.ndarray: New gradient
        """
        return gradient
