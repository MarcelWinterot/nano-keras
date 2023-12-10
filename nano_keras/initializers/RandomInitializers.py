import numpy as np
from nano_keras.initializers import RandomInitializer


class RandomNormal(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """_summary_

        Args:
            shape (tuple): _description_
            fan_in (int): _description_
            datatype (np.float_): _description_

        Returns:
            np.ndarray: _description_
        """
        return np.random.normal(size=shape).astype(datatype)


class RandomUniform(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """_summary_

        Args:
            shape (tuple): _description_
            fan_in (int): _description_
            datatype (np.float_): _description_

        Returns:
            np.ndarray: _description_
        """
        return np.random.uniform(self.mu, self.sigma, shape).astype(datatype)


class HeNormal(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """_summary_

        Args:
            shape (tuple): _description_
            fan_in (int): _description_
            datatype (np.float_): _description_

        Returns:
            np.ndarray: _description_
        """
        fan_in, _ = self.compute_fans(shape)
        output = np.random.normal(size=shape)
        output *= np.sqrt(2./fan_in)
        return output.astype(datatype)


class HeUniform(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """_summary_

        Args:
            shape (tuple): _description_
            fan_in (int): _description_
            datatype (np.float_): _description_

        Returns:
            np.ndarray: _description_
        """
        fan_in, _ = self.compute_fans(shape)
        limit = np.sqrt(6. / fan_in)
        output = np.random.uniform(-limit, limit, shape)
        return output.astype(datatype)


class XavierNormal(RandomInitializer):
    def __call__(self, shape: tuple, fan_in: int, datatype: np.float_) -> np.ndarray:
        fan_in, _ = self.compute_fans(shape)
        output = np.random.normal(size=shape)
        output = 2 * output - 1
        output *= np.sqrt(6 / fan_in)
        return output.astype(datatype)


class XavierUniform(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        fan_in, fan_out = self.compute_fans(shape)
        limit = np.sqrt(6. / (fan_in + fan_out))
        output = np.random.uniform(-limit, limit, shape)
        return output.astype(datatype)
