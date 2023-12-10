import numpy as np
from nano_keras.initializers import RandomInitializer


class RandomNormal(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Generates the parameters in the shape of 'shape' using the normal distribution.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Generated output using normal distribution
        """
        return np.random.normal(size=shape).astype(datatype)


class RandomUniform(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Generates the parameters in the shape of 'shape' using the uniform distribution.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Generated output using uniform distribution
        """
        return np.random.uniform(self.mu, self.sigma, shape).astype(datatype)


class HeNormal(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Generates the parameters in the shape of 'shape' using the he initalization strategy and normal distribution.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Generated output using he initalization strategy

        [He et al., 2015](https://arxiv.org/abs/1502.01852)
        """
        fan_in, _ = self.compute_fans(shape)
        output = np.random.normal(size=shape)
        output *= np.sqrt(2./fan_in)
        return output.astype(datatype)


class HeUniform(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Generates the parameters in the shape of 'shape' using the he initalization strategy and uniform distribution.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Generated output using he initalization strategy

        [He et al., 2015](https://arxiv.org/abs/1502.01852)
        """
        fan_in, _ = self.compute_fans(shape)
        limit = np.sqrt(6. / fan_in)
        output = np.random.uniform(-limit, limit, shape)
        return output.astype(datatype)


class XavierNormal(RandomInitializer):
    def __call__(self, shape: tuple, fan_in: int, datatype: np.float_) -> np.ndarray:
        """Generates the parameters in the shape of 'shape' using the Xavier/Gloriot initalization strategy and normal distribution.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Generated output using Xavier/Gloriot initalization strategy

        [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
        """
        fan_in, _ = self.compute_fans(shape)
        output = np.random.normal(size=shape)
        output = 2 * output - 1
        output *= np.sqrt(6 / fan_in)
        return output.astype(datatype)


class XavierUniform(RandomInitializer):
    def __call__(self, shape: tuple, datatype: np.float_) -> np.ndarray:
        """Generates the parameters in the shape of 'shape' using the Xavier/Gloriot initalization strategy and uniform distribution.

        Args:
            shape (tuple): Shape of the output
            datatype (np.float_): Datatype in which you want to store the data in.

        Returns:
            np.ndarray: Generated output using Xavier/Gloriot initalization strategy

        [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
        """
        fan_in, fan_out = self.compute_fans(shape)
        limit = np.sqrt(6. / (fan_in + fan_out))
        output = np.random.uniform(-limit, limit, shape)
        return output.astype(datatype)
