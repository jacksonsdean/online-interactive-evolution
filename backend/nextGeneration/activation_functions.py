"""Contains activation functions for nodes of the CPPN."""
import math
import numpy as np
from scipy.special import expit
from scipy import signal

def identity(x):
    """Returns the input value."""
    return x

def sigmoid(x):
    """Returns the sigmoid of the input."""
    return expit(x)

def tanh(x):
    """Returns the hyperbolic tangent of the input."""
    y = np.tanh(2.5*x)
    return y

def sawtooth(x):
    """Returns the sawtooth of the input."""
    return signal.sawtooth(x*10)

def tanh_sig(x):
    """Returns the sigmoid of the hyperbolic tangent of the input."""
    return sigmoid(tanh(x))

def sin(x):
    """Returns the sine of the input."""
    y =  np.sin(x*math.pi)
    return y

def cos(x):
    """Returns the cosine of the input."""
    y =  np.cos(x*math.pi)
    return y

def gauss(x):
    """Returns the gaussian of the input."""
    y = 2*np.exp(-20.0 * (x) ** 2)-1
    return y
