"""Contains utility functions"""
import inspect
import random
import sys

def is_valid_connection(from_node,to_node, config):
    """
    Checks if a connection is valid.
    params:
        from_node: The node from which the connection originates.
        to_node: The node to which the connection connects.
        config: The settings to check against
    returns:
        True if the connection is valid, False otherwise.
    """
    if from_node.layer == to_node.layer:
        return False  # don't allow two nodes on the same layer to connect

    if not config.allow_recurrent and from_node.layer > to_node.layer:
        return False  # invalid

    return True


def name_to_fn(name)->callable:
    """
    Converts a string to a function.
    params:
        name: The name of the function.
    returns:
        The function.
    """
    assert isinstance(name, str), f"name must be a string but is {type(name)}"
    fns = inspect.getmembers(sys.modules["activation_functions"])
    return fns[[f[0] for f in fns].index(name)][1]


def choose_random_function(config)->callable:
    """Chooses a random activation function from the activation function module."""
    random_fn = random.choice(config.activations)
    return  random_fn
