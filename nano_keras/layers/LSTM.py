import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.activations import Activation, ACTIVATIONS
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer
from nano_keras.initializers import Initializer, INITIALIZERS


class LSTM(LayerWithParams):
    def __init__(self, units: int, activation: Activation | str = "sigmoid", recurrent_activation: Activation | str = "tanh", weight_initalization: Initializer | str = "random_normal", recurrent_weight_initalization: Initializer | str = "random_normal", bias_initalization: Initializer | str = "zeros", return_sequences: bool = True, regulizer: Regularizer = None, name: str = "LSTM") -> None:
        self.units: int = units
        self.activation: Activation = activation if type(
            activation) == Activation else ACTIVATIONS[activation]
        self.recurrent_activation: Activation = recurrent_activation if type(
            recurrent_activation) == Activation else ACTIVATIONS[recurrent_activation]

        self.weight_initialization: Initializer = weight_initalization if type(
            weight_initalization) == Initializer else INITIALIZERS[weight_initalization]
        self.recurrent_weight_initalization: Initializer = recurrent_weight_initalization if type(
            recurrent_weight_initalization) == Initializer else INITIALIZERS[recurrent_weight_initalization]
        self.bias_initialization: Initializer = bias_initalization if type(
            bias_initalization) == Initializer else INITIALIZERS[bias_initalization]

        self.return_sequences: bool = return_sequences
        self.regulizer: Regularizer = regulizer
        self.name: str = name

    def output_shape(self, layers: list[Layer], current_layer_index: int) -> tuple:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)
        self.output_shape_value = (
            input_shape[0], self.units) if self.return_sequences else self.units

        return self.output_shape_value

    def __repr__(self) -> str:
        formatted_output = f"(None, {self.output_shape_value})"
        if type(self.output_shape_value) == tuple:
            formatted_output = f'(None, {", ".join(map(str, self.output_shape_value))})'

        return f"{self.name} (LSTM){' ' * (28 - len(self.name) - 6)}{formatted_output}{' ' * (26 - len(formatted_output))}{self.input_weights.size + self.recurrent_weights.size + self.biases.size}\n"

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_, bias_data_type: np.float_) -> None:
        input_shape = layers[current_layer_index -
                             1].output_shape(layers, current_layer_index-1)

        input_weights_shape = (input_shape[1], self.units)
        recurrent_weights_shape = (self.units, self.units)

        self.input_weights = np.random.randn(
            4, *input_weights_shape).astype(weight_data_type)
        self.recurrent_weights = np.random.randn(
            4, *recurrent_weights_shape).astype(weight_data_type)

        self.biases = np.random.randn(4, self.units)

        self.hidden_state = np.zeros((input_shape[0], self.units))
        self.cell_state = np.zeros((input_shape[0], self.units))

        self.output_shape_value = (
            input_shape[0], self.units) if self.return_sequences else self.units
        return

    def __call__(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.inputs = x

        if len(x.shape) != 2:
            raise ValueError(
                f"Input shape in LSTM layer must be 2d, received: {x.shape}")

        extra_dim = np.zeros((1, self.hidden_state.shape[1]))

        self.hidden_state = np.vstack((extra_dim, self.hidden_state))
        self.cell_state = np.vstack((extra_dim, self.cell_state))

        self.forget_gate = np.ndarray((x.shape[0], self.units))
        self.input_gate = np.ndarray((x.shape[0], self.units))
        self.candidate_cell_state = np.zeros((x.shape[0] + 1, self.units))
        self.output_gate = np.ndarray((x.shape[0], self.units))

        for time_stamp in range(1, x.shape[0]+1):
            # Note that we are using [time_stamp - 1] in most cases as otherwise we'd get an index of -1 at the first iteration
            # And for elements like hidden state and cell state we assign it to the current timestamp and the remove the first element

            # fₜ = σ(Wᵢ₁xₜ + Wᵣ₁hₜ₋₁ + b₁)
            self.forget_gate[time_stamp-1] = self.activation.apply_activation(
                np.dot(self.input_weights[0].T, x[time_stamp-1]) + np.dot(self.recurrent_weights[0], self.hidden_state[time_stamp-1]) + self.biases[0])

            # iₜ = σ(Wᵢ₂xₜ + Wᵣ₂hₜ₋₁ + b₂)
            self.input_gate[time_stamp-1] = self.activation.apply_activation(np.dot(
                self.input_weights[1].T, x[time_stamp-1]) + np.dot(self.recurrent_weights[1], self.hidden_state[time_stamp-1]) + self.biases[1])

            # C'ₜ = tanh(Wᵢ₃xₜ + Wᵣ₃hₜ₋₁ + b₃)
            self.candidate_cell_state[time_stamp] = self.recurrent_activation.apply_activation(np.dot(
                self.input_weights[2].T, x[time_stamp-1]) + np.dot(self.recurrent_weights[2], self.hidden_state[time_stamp-1]) + self.biases[2])

            # Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C'ₜ₋₁
            self.cell_state[time_stamp] = self.forget_gate[time_stamp-1] * self.cell_state[time_stamp-1] + \
                self.input_gate[time_stamp-1] * \
                self.candidate_cell_state[time_stamp-1]

            # oₜ = σ(Wᵢ₄xₜ + Wᵣ₄hₜ₋₁ + b₄)
            self.output_gate[time_stamp-1] = self.activation.apply_activation(np.dot(
                self.input_weights[3].T, x[time_stamp-1]) + np.dot(self.recurrent_weights[3], self.hidden_state[time_stamp-1]) + self.biases[3])

            # hₜ = oₜ ⊙ tanh(Cₜ)
            self.hidden_state[time_stamp] = self.output_gate[time_stamp-1] * \
                self.recurrent_activation.apply_activation(
                    self.cell_state[time_stamp])

        self.hidden_state = self.hidden_state[1:]
        self.cell_state = self.cell_state[1:]
        self.candidate_cell_state = self.candidate_cell_state[1:]

        if self.return_sequences:
            return self.hidden_state

        return self.hidden_state[-1]

    def backpropagate(self, gradient: np.ndarray, optimizer: Optimizer | list[Optimizer]) -> np.ndarray:
        if self.regulizer:
            gradient = self.regulizer.update_gradient(
                gradient, self.weights, self.biases)

        if len(gradient.shape) == 1:
            gradient = np.tile(gradient, (self.inputs.shape[0], 1))

        hidden_error = np.zeros_like(self.hidden_state)
        d_x = np.zeros_like(self.inputs)

        d_hidden_state = np.zeros_like(self.hidden_state)
        d_cell_state = np.zeros(
            (self.cell_state.shape[0] + 1, self.cell_state.shape[1]))
        d_cell_candidate = np.zeros_like(self.cell_state)

        d_input_gate = np.zeros_like(self.input_gate)
        d_forget_gate = np.zeros_like(self.forget_gate)
        d_output_gate = np.zeros_like(self.output_gate)

        d_gates = np.array(
            [d_cell_candidate, d_input_gate, d_forget_gate, self.output_gate])

        for time_stamp in range(self.inputs.shape[0])[-1::-1]:
            # δhₜ = Δt + Δhₜ
            d_hidden_state[time_stamp] = gradient[time_stamp] + \
                hidden_error[time_stamp]

            # δCₜ = δhₜ ⊙ oₜ ⊙ (1 - tanh²(Cₜ)) + δCₜ₊₁ ⊙ fₜ₊₁
            d_cell_state[time_stamp] * self.output_gate[time_stamp] * \
                self.recurrent_activation.compute_derivative(
                self.cell_state[time_stamp]) + d_cell_state[time_stamp+1] * self.forget_gate[time_stamp]

            # δC' = δCₜ ⊙ iₜ ⊙ (1-C'ₜ²)
            d_cell_candidate[time_stamp] = d_cell_candidate[time_stamp] * \
                self.input_gate[time_stamp] * \
                (1 - self.candidate_cell_state[time_stamp]**2)

            # δiₜ = δCₜ ⊙ C'ₜ ⊙ iₜ ⊙ (1-iₜ)
            d_input_gate[time_stamp] = d_cell_state[time_stamp] * self.candidate_cell_state[time_stamp] * \
                self.input_gate[time_stamp] * (1 - self.input_gate[time_stamp])

            # δfₜ = δCₜ ⊙ Cₜ₋₁ ⊙ fₜ ⊙ (1-fₜ)
            d_forget_gate[time_stamp] = d_cell_state[time_stamp] * self.candidate_cell_state[time_stamp] * \
                self.forget_gate[time_stamp] * \
                (1 - self.forget_gate[time_stamp])

            # δoₜ = δhₜ ⊙ tanh(Cₜ) ⊙ oₜ ⊙ (1-oₜ)
            d_output_gate[time_stamp] = d_hidden_state[time_stamp] * self.recurrent_activation.apply_activation(
                self.cell_state[time_stamp]) * self.output_gate[time_stamp] * (1 - self.output_gate[time_stamp])

            # Δhₜ₋₁ = Uᵗ * δgatesₜ
            hidden_error[time_stamp-1] = np.average(np.dot(
                self.recurrent_weights.T, d_gates[:, time_stamp]), axis=(1, 2))

            # δxₜ = Wᵗ * δgatesₜ
            d_x[time_stamp] = np.dot(self.input_weights.T,
                                     d_gates[:, time_stamp])[-1, :, -1]

        # Weights update
        delta_input_weights = np.ndarray(self.inputs.shape)
        delta_recurrent_weights = np.ndarray(
            d_gates.shape).transpose(1, 0, 2)

        for time_stamp in range(self.inputs.shape[0])[-1::-1]:
            # delta_input_weights[time_stamp] = d_gates[:,
            #    time_stamp] * self.inputs[time_stamp]

            delta_recurrent_weights[time_stamp] = d_gates[:,
                                                          time_stamp] * self.hidden_state[time_stamp]

        delta_recurrent_weights = delta_recurrent_weights.transpose(1, 0, 2)
        delta_biases = np.sum(d_gates, 1)

        print(f"input_weights: {self.input_weights.shape}, R_weights: {self.recurrent_weights.shape}, r_weights_delta: {delta_recurrent_weights.shape}, biases: {self.biases.shape}, delta_biases: {delta_biases.shape}")

        return d_x
