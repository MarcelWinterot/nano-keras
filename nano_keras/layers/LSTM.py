import numpy as np
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.activations import Activation, ACTIVATIONS
from nano_keras.optimizers import Optimizer
from nano_keras.regulizers import Regularizer


class LSTM(LayerWithParams):
    def __init__(self, units: int, activation: Activation | str = "sigmoid", recurrent_activation: Activation | str = "tanh", weight_initalization: str = "random", recurrent_weight_initalization: str = "random", return_sequences: bool = True, regulizer: Regularizer = None, name: str = "LSTM") -> None:
        self.units: int = units
        self.activation: Activation = activation if type(
            activation) == Activation else ACTIVATIONS[activation]
        self.recurrent_activation: Activation = recurrent_activation if type(
            recurrent_activation) == Activation else ACTIVATIONS[recurrent_activation]
        self.weight_initialization: str = weight_initalization
        self.recurrent_weight_initalization: str = recurrent_weight_initalization
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

    def generate_weights(self, layers: list[Layer], current_layer_index: int, weight_data_type: np.float_) -> None:
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
        raise NotImplementedError(
            f"LSTM backpropagation is not implemented yet. Please be patient")
    
        if self.regulizer:
            gradient = self.regulizer.update_gradient(
                gradient, self.weights, self.biases)

        self.hidden_error = np.zeros_like(self.hidden_state)

        self.d_hidden_state = np.zeros_like(self.hidden_state)
        self.d_cell_state = np.zeros(
            (self.cell_state.shape[0] + 1, self.cell_state.shape[1]))
        self.d_cell_candidate = np.zeros_like(self.cell_state)

        self.d_input_gate = np.zeros_like(self.input_gate)
        self.d_forget_gate = np.zeros_like(self.forget_gate)
        self.d_output_gate = np.zeros_like(self.output_gate)

        self.d_gates = np.array(
            [self.d_cell_candidate, self.d_input_gate, self.d_forget_gate, self.output_gate])

        for time_stamp in range(self.inputs.shape[0])[-1::-1]:
            # δhₜ = Δt + Δhₜ
            self.d_hidden_state[time_stamp] = gradient * \
                self.hidden_error[time_stamp]

            # δCₜ = δhₜ ⊙ oₜ ⊙ (1 - tanh²(Cₜ)) + δCₜ₊₁ ⊙ fₜ₊₁
            self.d_cell_state[time_stamp] * self.output_gate[time_stamp] * \
                self.recurrent_activation.compute_derivative(
                self.cell_state[time_stamp]) + self.d_cell_state[time_stamp+1] * self.forget_gate[time_stamp]

            # δC' = δCₜ ⊙ iₜ ⊙ (1-C'ₜ²)
            self.d_cell_candidate[time_stamp] = self.d_cell_candidate[time_stamp] * \
                self.input_gate[time_stamp] * \
                (1 - self.candidate_cell_state[time_stamp]**2)

            # δiₜ = δCₜ ⊙ C'ₜ ⊙ iₜ ⊙ (1-iₜ)
            self.d_input_gate[time_stamp] = self.d_cell_state[time_stamp] * self.candidate_cell_state[time_stamp] * \
                self.input_gate[time_stamp] * (1 - self.input_gate[time_stamp])

            # δfₜ = δCₜ ⊙ Cₜ₋₁ ⊙ fₜ ⊙ (1-fₜ)
            self.d_forget_gate[time_stamp] = self.d_cell_state[time_stamp] * self.candidate_cell_state[time_stamp] * \
                self.forget_gate[time_stamp] * \
                (1 - self.forget_gate[time_stamp])

            # δoₜ = δhₜ ⊙ tanh(Cₜ) ⊙ oₜ ⊙ (1-oₜ)
            self.d_output_gate[time_stamp] = self.d_hidden_state * self.recurrent_activation.apply_activation(
                self.cell_state[time_stamp]) * self.output_gate[time_stamp] * (1 - self.output_gate[time_stamp])

            self.hidden_error[time_stamp -
                              1] = self.recurrent_weights * self.d_gates
