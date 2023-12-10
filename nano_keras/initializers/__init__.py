from nano_keras.initializers.Initializers import Initializer, RandomInitializer, ConstantInitializer
from nano_keras.initializers.RandomInitializers import RandomNormal, RandomUniform, HeNormal, HeUniform, XavierNormal
from nano_keras.initializers.ConstantInitializer import Ones, Zeros, Constant

INITIALIZERS = {"random_normal": RandomNormal(
), "xavier_normal": XavierNormal(), "he_normal": HeNormal(), "random_uniform": RandomUniform(), "he_uniform": HeUniform(), "ones": Ones(), "zeros": Zeros()}
