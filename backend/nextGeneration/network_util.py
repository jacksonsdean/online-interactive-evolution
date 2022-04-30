"""Contains utility functions"""
import inspect
import random
import sys
try:
    import nextGeneration.activation_functions as af
except ModuleNotFoundError:
    import activation_functions as af


def is_valid_connection(from_node, to_node, config):
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


def name_to_fn(name) -> callable:
    """
    Converts a string to a function.
    params:
        name: The name of the function.
    returns:
        The function.
    """
    assert isinstance(name, str), f"name must be a string but is {type(name)}"
    if name == "":
        return None
    fns = inspect.getmembers(sys.modules[af.__name__])
    return fns[[f[0] for f in fns].index(name)][1]


def choose_random_function(config) -> callable:
    """Chooses a random activation function from the activation function module."""
    random_fn = random.choice(config.activations)
    return random_fn


def get_disjoint_connections(this_cxs, other_innovation):
    """returns connections in this_cxs that do not share an innovation number with a
        connection in other_innovation"""
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if (t_cx.innovation not in other_innovation and t_cx.innovation < other_innovation[-1])]


def get_excess_connections(this_cxs, other_innovation):
    """returns connections in this_cxs that share an innovation number with a
        connection in other_innovation"""
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if (t_cx.innovation not in other_innovation and t_cx.innovation > other_innovation[-1])]

def get_matching_connections(cxs_1, cxs_2):
    """returns connections in cxs_1 that share an innovation number with a connection in cxs_2
       and     connections in cxs_2 that share an innovation number with a connection in cxs_1"""
    return sorted([c1 for c1 in cxs_1 if c1.innovation in [c2.innovation for c2 in cxs_2]], key=lambda x: x.innovation),\
        sorted([c2 for c2 in cxs_2 if c2.innovation in [
               c1.innovation for c1 in cxs_1]], key=lambda x: x.innovation)

def find_node_with_id(nodes, id):
    """Returns the node with the given id from the list of nodes"""
    for node in nodes:
        if node.id == id:
            return node
    return None
