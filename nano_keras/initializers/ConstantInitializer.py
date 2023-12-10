import numpy as np
from nano_keras.initializers import ConstantInitializer


class Zeros(ConstantInitializer):
    def __init__(self) -> None:
        pass

    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Computes an output in the shape of 'shape' filled with zeros.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Output in the shape of 'shape' fileld with zeros
        """
        return np.ones(shape, datatype)


class Ones(ConstantInitializer):
    def __init__(self) -> None:
        pass

    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Computes an output in the shape of 'shape' filled with ones.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Output in the shape of 'shape' fileld with ones
        """
        return np.ones(shape, datatype)


class Constant(ConstantInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Computes an output in the shape of 'shape' filled with 'constant' specified in the __init__.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Output in the shape of 'shape' fileld with 'constant' specified in the __init__
        """
        output = np.ndarray(shape, datatype)
        output.fill(self.constant)
        return output
