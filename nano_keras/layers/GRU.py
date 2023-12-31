import numpy as np
from nano_keras.activations import Activation, ACTIVATIONS
from nano_keras.initializers import Initializer, INITIALIZERS
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer


class GRU(LayerWithParams):
    """GRU layer. It's a gated recurrent unit layer. It's input shape is (None, input_length, input_dim) and it's output shape is (None, input_length, output_dim)
    """

    def __init__(self, units: int, activation: Activation | str = "tanh", recurrent_actvation: Activation | str = "sigmoid", weight_initialization: Initializer | str = "random_normal", recurrent_weight_initialization: Initializer | str = "random_normal", bias_initalization: Initializer | str = "zeros", return_sequences: bool = True, regulizer: Regularizer = None, trainable: bool = True, input_shape: tuple = None, name: str = "GRU") -> None:
        """GRU layer initializer

        Args:
            units (int): Number of gates/neurons the layer should have
            activation (Activation | str, optional): Activation function the current_memory_content should use. Defaults to "tanh".
            recurrent_actvation (Activation | str, optional): Activation function the rest of the gates should use. Defaults to "sigmoid".
            weight_initaliziton (str, optional): Intialization strategy you want to use to generate input weights of the layer. You can find all of them in the Initalizers folder. Defalut to "random_normal"
            recurrent_weight_initaliziton (str, optional): Intialization strategy you want to use to generate recurrent weights of the layer. You can find all of them in the Initalizers folder. Defalut to "random_normal"
            bias_initialization (Initalizer | str, optional): Intialization strategy you want to use to generate biases of the layer. You can find all of them in the Initalizers folder. Defalut to "random_normal"
            return_sequences (bool, optional): Should the layer return all timestamps or only the last one. True - all, False - last timestamp. Defaults to True.
            regulizer (Regularizer, optional): Regulizer the model should use. You can find them all in the regulizers.py file. You must pass the already intialized class. Defaults to None.
            trainable (bool, optional): Parameter that decides whether the parameters should be updated or no. Defaults to True.
            input_shape (tuple, optional): Input shape to the layer. Used if you dont't want to use Input layer. If it's None it won't be used. Defaults to None.
            name (str, optional): Name of the layer. Helpful for debugging. Defaults to "Layer".
        """
        self.units: int = units
        self.activation: Activation = activation if type(
            activation) == Activation else ACTIVATIONS[activation]
        self.recurrent_activation: Activation = recurrent_actvation if type(
            recurrent_actvation) == Activation else ACTIVATIONS[recurrent_actvation]

        self.weight_initialization: Initializer = weight_initialization if type(
            weight_initialization) == Initializer else INITIALIZERS[weight_initialization]
        self.recurrent_weight_initialization: Initializer = recurrent_weight_initialization if type(
            recurrent_weight_initialization) == Initializer else INITIALIZERS[recurrent_weight_initialization]
        self.bias_initialization: Initializer = bias_initalization if type(
            bias_initalization) == Initializer else INITIALIZERS[bias_initalization]

        self.return_sequences: bool = return_sequences
        self.regulizer: Regularizer = regulizer
        self.trainable: bool = trainable
        self.input_shape: tuple = input_shape
        self.name: str = name

        self.current_batch: int = 0

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1) if self.input_shape is None else self.input_shape

        self.output_shape_value = (
            input_shape[0], self.units) if self.return_sequences else self.units

        return self.output_shape_value

    def set_batch_size(self, batch_size: int, layers: list, index: int) -> None:
        self.batch_size = batch_size

        input_shape = layers[index-1].output_shape(
            layers, index-1) if self.input_shape is None else self.input_shape
        input_shape = tuple(input_shape)

        self.inputs = np.zeros((batch_size, *input_shape))
        self.hidden_state = np.zeros(
            (batch_size, input_shape[0] + 1, self.units))

        self.update_gate = np.zeros((batch_size, input_shape[0], self.units))
        self.reset_gate = np.zeros((batch_size, input_shape[0], self.units))
        self.current_memory_content = np.zeros(
            (batch_size, input_shape[0] + 1, self.units))

    def __repr__(self) -> str:
        formatted_output = f"(None, {self.output_shape_value})"
        if type(self.output_shape_value) == tuple:
            formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'

        return f"{self.name} (GRU){' ' * (28 - len(self.name) - 5)}{formatted_output}{' ' * (26 - len(formatted_output))}{self.input_weights.size + self.recurrent_weights.size + self.biases.size}\n"

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_, bias_data_type: np.float_) -> None:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1) if self.input_shape is None else self.input_shape

        input_weights_shape = (3, input_shape[1], self.units)
        recurrent_weights_shape = (3, self.units, self.units)

        self.input_weights = self.weight_initialization(
            input_weights_shape, weight_data_type)
        self.recurrent_weights = self.recurrent_weight_initialization(
            recurrent_weights_shape, weight_data_type)

        self.biases = self.bias_initialization(
            (2, 3, self.units), bias_data_type)

        self.output_shape_value = (
            input_shape[0], self.units) if self.return_sequences else self.units

    def get_number_of_params(self) -> tuple:
        if self.trainable:
            return (self.input_weights.size + self.recurrent_weights.size + self.biases.size, 0)
        return (0, self.input_weights.size + self.recurrent_weights.size + self.biases.size)

    def get_params_size(self) -> tuple:
        if self.trainable:
            return (self.input_weights.nbytes + self.recurrent_weights.nbytes + self.biases.nbytes, 0)
        return (0, self.input_weights.nbytes + self.recurrent_weights.nbytes + self.biases.nbytes)

    def get_weights(self) -> list[np.ndarray, np.ndarray]:
        return [self.input_weights, self.recurrent_weights, self.biases]

    def set_weights(self, input_weights: np.ndarray, recurrent_weights: np.ndarray, biases: np.ndarray) -> None:
        self.input_weights = input_weights
        self.recurrent_weights = recurrent_weights
        self.biases = biases

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        if len(x.shape) != 2:
            raise ValueError(
                f"Input shape in GRU layer must be 2d, received: {x.shape}")

        self.inputs[self.current_batch] = x

        for time_stamp in range(1, x.shape[0]+1):
            self.update_gate[self.current_batch, time_stamp-1] = self.recurrent_activation.apply_activation(
                np.dot(self.input_weights[0].T, x[time_stamp-1]) + np.dot(self.recurrent_weights[0],
                                                                          self.hidden_state[self.current_batch, time_stamp-1]) + self.biases[0, 0] + self.biases[1, 0])

            self.reset_gate[self.current_batch, time_stamp-1] = self.recurrent_activation.apply_activation(
                np.dot(self.input_weights[0].T, x[time_stamp-1]) + np.dot(self.recurrent_weights[0],
                                                                          self.hidden_state[self.current_batch, time_stamp-1]) + self.biases[0, 0] + self.biases[1, 0])

            self.current_memory_content[self.current_batch, time_stamp] = self.activation.apply_activation(
                np.dot(self.input_weights[2].T, x[time_stamp-1]) + np.dot(self.recurrent_weights[2],
                                                                          (self.hidden_state[self.current_batch, time_stamp-1] * self.reset_gate[self.current_batch, time_stamp-1])) + self.biases[0, 2] + self.biases[1, 2])

            self.hidden_state[self.current_batch, time_stamp] = self.update_gate[self.current_batch, time_stamp-1] * self.hidden_state[self.current_batch, time_stamp - 1] + (
                1 - self.update_gate[self.current_batch, time_stamp-1]) * self.current_memory_content[self.current_batch, time_stamp]

        index = self.current_batch
        if is_training:
            self.current_batch += 1

        if self.return_sequences:
            return self.hidden_state[index, 1:]

        return self.hidden_state[index, -1]

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        """Backpropagate algorithm used for GRU layer.

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

        if len(gradient.shape) == 1:
            gradient = np.tile(gradient, (self.inputs.shape[0], 1))

        update_gate_gradient = np.ndarray((gradient.shape[0], self.units))
        reset_gate_gradient = np.ndarray((gradient.shape[0], self.units))
        current_memory_content_gradient = np.ndarray(
            (gradient.shape[0], self.units))

        input_weights_gradient = np.ndarray(self.input_weights.shape)
        recurrent_weights_gradient = np.ndarray(
            self.recurrent_weights.shape)
        biases_gradient = np.ndarray(self.biases.shape)

        inputs = np.average(self.inputs, axis=0)
        current_memory_content = np.average(
            self.current_memory_content[:, 1:], axis=0)
        hidden_state = np.average(self.hidden_state[:, 1:], axis=0)
        update_gate = np.average(self.update_gate, axis=0)
        reset_gate = np.average(self.reset_gate, axis=0)

        for time_stamp in range(gradient.shape[0]-1, -1, -1):
            update_gate_gradient[time_stamp] = gradient[time_stamp] * (
                current_memory_content[time_stamp] - hidden_state[time_stamp])

            current_memory_content_gradient[time_stamp] = gradient[time_stamp] * \
                update_gate[time_stamp]

            reset_gate_gradient[time_stamp] = gradient[time_stamp] * \
                hidden_state[time_stamp] * \
                current_memory_content[time_stamp]

            gate_gradients = [update_gate_gradient[time_stamp], reset_gate_gradient[time_stamp],
                              current_memory_content_gradient[time_stamp]]

            for i, gate_gradient in enumerate(gate_gradients):
                input_weights_gradient[i] += np.outer(
                    inputs[time_stamp], gate_gradient)

                recurrent_weights_gradient[i] += np.outer(
                    hidden_state[time_stamp-1], gate_gradient)

                biases_gradient[:, i] += gate_gradient[i]

        # Input weights and biases
        if self.trainable:
            self.input_weights, self.biases[0] = optimizer[0].apply_gradients(
                input_weights_gradient, biases_gradient[0], self.input_weights, self.biases[0])

            self.recurrent_weights, self.biases[1] = optimizer[0].apply_gradients(
                recurrent_weights_gradient, biases_gradient[1], self.recurrent_weights, self.biases[1])

        self.current_batch = 0

        return np.dot(update_gate_gradient, self.input_weights[0].T) + np.dot(reset_gate_gradient, self.input_weights[1].T) + np.dot(current_memory_content_gradient, self.input_weights[2].T)
