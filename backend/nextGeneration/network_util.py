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
    fns.extend([("", None)])
    return fns[[f[0] for f in fns].index(name)][1]


def choose_random_function():
    """Chooses a random activation function from the activation function module."""
    fns = inspect.getmembers(sys.modules["activation_functions"])
    fns.extend([("", None)])
    random_fn = random.choice(fns)

    return  name_to_fn(random_fn[0])
