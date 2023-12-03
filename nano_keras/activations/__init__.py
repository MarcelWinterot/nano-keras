from nano_keras.activations.Activation import Activation
from nano_keras.activations.ELU import ELU
from nano_keras.activations.LeakyReLU import LeakyReLU
from nano_keras.activations.ReLU import ReLU
from nano_keras.activations.Sigmoid import Sigmoid
from nano_keras.activations.Softmax import Softmax
from nano_keras.activations.Tanh import Tanh

ACTIVATIONS = {'sigmoid': Sigmoid(), 'tanh': Tanh(), 'relu': ReLU(
), 'leaky_relu': LeakyReLU(), 'elu': ELU(), "softmax": Softmax()}
