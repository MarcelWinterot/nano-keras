import numpy as np


class Initializer:
    def __init__(self) -> None:
        """Initializer for base class Initializer class
        """
        pass

    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Base implementation of generating parameters for Initializer class

        Args:
            shape (tuple): Shape of the parameters you want to generate
            datatype (np.float_): Data type you want to use to store the parameters, for example np.float32 or np.float64

        Returns:
            np.ndarray: Generated parameters
        """
        pass


class RandomInitializer(Initializer):
    def __init__(self, mu: float = 0, sigma: float = 0.1) -> None:
        """Initalizer for all the initializers which use some randomness in them like Random Normal, He Uniform, Xavier Normal, etc.

        Args:
            mu (float, optional): mean of the uniform distribution. Defaults to 0.
            sigma (float, optional): standard deviation, half-width of the distribution. Defaults to 0.1.
            The generated number will be between mu and mu + selfsigma.
        """
        self.mu = mu
        self.sigma = sigma

    @staticmethod
    def compute_fans(shape: tuple) -> tuple:
        """Computes the number of input and output units for a weight shape.

        Args:
            shape: Integer shape tuple.

        Returns:
            A tuple of integer scalars: `(fan_in, fan_out)`.
        """
        shape = tuple(shape)
        if len(shape) < 1:
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            receptive_field_size = 1
            for dim in shape[:-2]:
                receptive_field_size *= dim
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        return int(fan_in), int(fan_out)


class ConstantInitializer(Initializer):
    def __init__(self, constant: float) -> None:
        """Initializer for all Initializers which use a constant variable as the parameter.

        Args:
            constant (float): The value you want the parameters to be set to.
        """
        self.constant = constant
