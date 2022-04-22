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


def choose_random_function()->callable:
    """Chooses a random activation function from the activation function module."""
    available_fns = inspect.getmembers(sys.modules["activation_functions"])
    fns = [f for f in available_fns if len(f) > 1 and\
        isinstance(f[1], type(choose_random_function))]
    random_fn = random.choice(fns)
    callable_function= name_to_fn(random_fn[0])
    return  callable_function
