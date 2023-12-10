import numpy as np
from nano_keras.initializers import ConstantInitializer


class Zeros(ConstantInitializer):
    def __init__(self) -> None:
        pass

    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        return np.ones(shape, datatype)


class Ones(ConstantInitializer):
    def __init__(self) -> None:
        pass

    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        return np.ones(shape, datatype)


class Constant(ConstantInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        output = np.ndarray(shape, datatype)
        output.fill(self.constant)
        return output
