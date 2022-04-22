"""Contains utility functions"""
import inspect
import random
import sys


def name_to_fn(name)->callable:
    """
    Converts a string to a function.
    params:
        name: The name of the function.
    returns:
        The function.
    """
    fns = inspect.getmembers(sys.modules["activation_functions"])
    return fns[[f[0] for f in fns].index(name)][1]


def choose_random_function(config)->callable:
    """Chooses a random activation function from the activation function module."""
    random_fn = random.choice(config.activations)
    return  random_fn
