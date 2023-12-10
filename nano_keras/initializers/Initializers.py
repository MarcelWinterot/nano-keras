import numpy as np


class Initializer:
    def __init__(self) -> None:
        """_summary_
        """
        pass

    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """_summary_

        Args:
            shape (tuple): _description_
            fan_in (int): _description_
            datatype (np.float_): _description_

        Returns:
            np.ndarray: _description_
        """
        pass


class RandomInitializer(Initializer):
    def __init__(self, mu: float = 0, sigma: float = 0.1) -> None:
        """_summary_

        Args:
            mu (float, optional): _description_. Defaults to 0.
            sigma (float, optional): _description_. Defaults to 0.1.
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
        """_summary_

        Args:
            constant (float): _description_
        """
        self.constant = constant
