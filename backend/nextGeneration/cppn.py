"""Contains the CPPN, Node, and Connection classes."""
import copy
from enum import IntEnum
import math
import json
import numpy as np

try:
    from nextGeneration.activation_functions import identity
    from nextGeneration.graph_util import name_to_fn, choose_random_function, is_valid_connection
    from nextGeneration.graph_util import get_matching_connections, find_node_with_id, feed_forward_layers
except ModuleNotFoundError:
    from activation_functions import identity
    from graph_util import get_matching_connections, find_node_with_id, feed_forward_layers
    from graph_util import name_to_fn, choose_random_function, is_valid_connection

class NodeType(IntEnum):
    """Enum for the type of node."""
    INPUT  = 0
    OUTPUT = 1
    HIDDEN = 2


class Node:
    """Represents a node in the CPPN."""

    @staticmethod
    def create_from_json(json_dict):
        """Constructs a node from a json dict or string."""
        i = Node(None, None, None, None)
        i = i.from_json(json_dict)
        return i

    @staticmethod
    def empty():
        """Returns an empty node."""
        return Node(identity, NodeType.HIDDEN, 0, 0)

    def __init__(self, activation, _type, _id, _layer=999) -> None:
        self.activation = activation
        self.id = _id
        self.type = _type
        self.layer = _layer
        self.sum_inputs = None
        self.outputs = None

    def to_json(self):
        """Converts the node to a json string."""
        self.type = int(self.type)
        self.id = int(self.id)
        self.layer = int(self.id)
        self.sum_inputs = np.array([]).tolist()
        self.outputs = np.array([]).tolist()
        try:
            self.activation = self.activation.__name__
        except AttributeError:
            pass
        return json.dumps(self.__dict__)

    def from_json(self, json_dict):
        """Constructs a node from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        self.__dict__ = json_dict
        self.type = NodeType(self.type)
        self.activation = name_to_fn(self.activation)
        self.outputs = None
        self.sum_inputs = None
        return self



class Connection:
    """
    Represents a connection between two nodes.
    connection            e.g.  2->5,  1->4
    innovation_number            0      1
    where innovation number is the same for all of same connection
    i.e. 2->5 and 2->5 have same innovation number, regardless of individual
    """
    innovations = []

    @staticmethod
    def get_innovation(from_node, to_node):
        """Returns the innovation number for the connection."""
        connection_from_to = (from_node.id, to_node.id) # based on id

        if connection_from_to in Connection.innovations:
            return Connection.innovations.index(connection_from_to)

        Connection.innovations.append(connection_from_to)
        return len(Connection.innovations) - 1

    def __init__(self, from_node, to_node, weight, enabled = True) -> None:
        # Initialize
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.innovation = Connection.get_innovation(from_node, to_node)
        self.enabled = enabled
        self.is_recurrent = to_node.layer < from_node.layer

    def to_json(self):
        """Converts the connection to a json string."""
        self.innovation = int(self.innovation)
        if isinstance(self.from_node, Node):
            self.from_node = self.from_node.to_json()
        if isinstance(self.to_node, Node):
            self.to_node = self.to_node.to_json()
        return json.dumps(self.__dict__)

    def from_json(self, json_dict):
        """Constructs a connection from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        self.__dict__ = json_dict
        self.from_node = Node.create_from_json(self.from_node)
        self.to_node = Node.create_from_json(self.to_node)
        return self

    @staticmethod
    def create_from_json(json_dict):
        """Constructs a connection from a json dict or string."""
        f_node = Node.empty()
        t_node = Node.empty()
        i = Connection(f_node, t_node, 0)
        i.from_json(json_dict)
        return i

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"([{self.from_node.id}->{self.to_node.id}]"+\
            f"I:{self.innovation})"
        # return f"([{self.from_node.id}->{self.to_node.id}]"+\
            # f"I:{self.innovation} W:{self.weight:3f} E:{self.enabled} R:{self.is_recurrent})"


class CPPN():
    """A CPPN Object with Nodes and Connections."""

    pixel_inputs = np.zeros((0,0))
    @staticmethod
    def initialize_inputs(res_h, res_w, use_radial_dist, use_bias, n_inputs):
        """Initializes the pixel inputs."""
        x_vals = np.linspace(-.5, .5, res_w)
        y_vals = np.linspace(-.5, .5, res_h)
        CPPN.pixel_inputs = np.zeros((res_h, res_w, n_inputs))
        for y in range(res_h):
            for x in range(res_w):
                this_pixel = [y_vals[y], x_vals[x]] # coordinates
                if use_radial_dist:
                    # d = sqrt(x^2 + y^2)
                    this_pixel.append(math.sqrt(y_vals[y]**2 + x_vals[x]**2))
                if use_bias:
                    this_pixel.append(1.0)# bias = 1.0
                CPPN.pixel_inputs[y][x] = this_pixel

    def __init__(self, config, nodes = None, connections = None) -> None:
        self.image = None
        self.node_genome = []  # inputs first, then outputs, then hidden
        self.connection_genome = []
        self.selected = False

        self.n_inputs = 2 # x, y
        if config.use_radial_distance:
            self.n_inputs += 1
        if config.use_input_bias:
            self.n_inputs+=1
        self.n_outputs = len(config.color_mode)

        self.config = config

        if nodes is None:
            self.initialize_node_genome()
        else:
            self.node_genome = nodes
        if connections is None:
            self.initialize_connection_genome()
        else:
            self.connection_genome = connections



    def initialize_connection_genome(self):
        """Initializes the connection genome."""
        output_layer = self.node_genome[self.n_inputs].layer

        for layer_index in range(0, output_layer):
            layer_from = self.get_layer(layer_index)
            for _, from_node in enumerate(layer_from):
                layer_to = self.get_layer(layer_index+1)
                for _, to_node in enumerate(layer_to):
                    new_cx = Connection(
                        from_node, to_node, self.random_weight())
                    self.connection_genome.append(new_cx)
                    if np.random.rand() > self.config.init_connection_probability:
                        new_cx.enabled = False


    def initialize_node_genome(self):
        """Initializes the node genome."""
        total_node_count = self.n_inputs + \
            self.n_outputs + self.config.hidden_nodes_at_start
        for _ in range(self.n_inputs):
            self.node_genome.append(
                Node(identity, NodeType.INPUT, self.get_new_node_id(), 0))
        for _ in range(self.n_inputs, self.n_inputs + self.n_outputs):
            if self.config.output_activation is None:
                output_fn = choose_random_function(self.config)
            else:
                output_fn = self.config.output_activation
            self.node_genome.append(
                Node(output_fn, NodeType.OUTPUT, self.get_new_node_id(), 2))
        for _ in range(self.n_inputs + self.n_outputs, total_node_count):
            self.node_genome.append(Node(choose_random_function(self.config), NodeType.HIDDEN,
                self.get_new_node_id(), 1))

    def to_json(self):
        """Converts the CPPN to a json string."""
        img = self.image
        if img is not None:
            img = list(img)
            img = json.dumps(img)
        # make copies to keep the CPPN intact
        copy_of_nodes = copy.deepcopy(self.node_genome)
        copy_of_connections = copy.deepcopy(self.connection_genome)
        return {"node_genome": [n.to_json() for n in copy_of_nodes], "connection_genome":\
            [c.to_json() for c in copy_of_connections], "image": img, "selected": self.selected}

    def from_json(self, json_dict):
        """Constructs a CPPN from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        for k, v in json_dict.items():
            setattr(self, k, v)
        for i, cx in enumerate(self.connection_genome):
            self.connection_genome[i] = Connection.create_from_json(cx)
        for i, n in enumerate(self.node_genome):
            self.node_genome[i] = Node.create_from_json(n)

        for cx in self.connection_genome:
            cx.from_node = find_node_with_id(self.node_genome, cx.from_node.id)
            cx.to_node = find_node_with_id(self.node_genome, cx.to_node.id)

        self.update_node_layers()
        CPPN.initialize_inputs(self.config.res_h, self.config.res_w,
                self.config.use_radial_distance,
                self.config.use_input_bias,
                self.n_inputs)

    @staticmethod
    def create_from_json(json_dict, config):
        """Constructs a CPPN from a json dict or string."""
        i = CPPN(config)
        i.from_json(json_dict)
        return i

    def random_weight(self):
        """Returns a random weight between -max_weight and max_weight."""
        return np.random.uniform(-self.config.max_weight, self.config.max_weight)

    def get_new_node_id(self):
        """Returns a new node id."""
        new_id = 0
        while len(self.node_genome) > 0 and new_id in [node.id for node in self.node_genome]:
            new_id += 1
        return new_id

    def enabled_connections(self):
        """Returns a yield of enabled connections."""
        for connection in self.connection_genome:
            if connection.enabled:
                yield connection

    def mutate_activations(self):
        """Mutates the activation functions of the nodes."""
        eligible_nodes = list(self.hidden_nodes())
        if self.config.output_activation is None:
            eligible_nodes.extend(self.output_nodes())
        if self.config.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes())
        for node in eligible_nodes:
            if np.random.uniform(0,1) < self.config.prob_mutate_activation:
                node.activation = choose_random_function(self.config)

    def mutate_weights(self):
        """
        Each connection weight is perturbed with a fixed probability by
        adding a floating point number chosen from a uniform distribution of
        positive and negative values """

        for connection in self.connection_genome:
            if np.random.uniform(0, 1) < self.config.prob_mutate_weight:
                connection.weight += np.random.uniform(-self.config.weight_mutation_max,
                                               self.config.weight_mutation_max)
            elif np.random.uniform(0, 1) < self.config.prob_weight_reinit:
                connection.weight = self.random_weight()

        self.clamp_weights()

    def mutate(self):
        """Mutates the CPPN based on its config."""
        if(np.random.uniform(0,1) < self.config.prob_add_node):
            self.add_node()
            print("Added node")
        if(np.random.uniform(0,1) < self.config.prob_remove_node):
            self.remove_node()
            print("removed node")
        if(np.random.uniform(0,1) < self.config.prob_add_connection):
            self.add_connection()
            print("Added connection")
        if(np.random.uniform(0,1) < self.config.prob_disable_connection):
            self.disable_connection()
            print("disabled connection")

        self.mutate_activations()
        self.mutate_weights()
        self.update_node_layers()
        self.disable_invalid_connections()

    def disable_invalid_connections(self):
        return
        """Disables connections that are not compatible with the current configuration."""
        for connection in self.connection_genome:
            if connection.enabled:
                if not is_valid_connection(connection.from_node, connection.to_node, self.config):
                    print(f"Disabling invalid connection {connection.from_node.id} -> {connection.to_node.id}")
                    connection.enabled = False


    def add_connection(self):
        """Adds a connection to the CPPN."""
        for _ in range(20):  # try 20 times
            [from_node, to_node] = np.random.choice(
                self.node_genome, 2, replace=False)
            existing_cx = None
            for cx in self.connection_genome:
                if cx.from_node == from_node and cx.to_node == to_node:
                    existing_cx = cx
            if existing_cx is not None:
                if not existing_cx.enabled and np.random.rand() < self.config.prob_reenable_connection:
                    existing_cx.enabled = True     # re-enable the connection
                break  # don't allow duplicates

            if is_valid_connection(from_node, to_node, self.config):
                # valid connection, add
                new_cx = Connection(from_node, to_node, self.random_weight())
                self.connection_genome.append(new_cx)
                self.update_node_layers()
                break

        # failed to find a valid connection, don't add

    def add_node(self):
        """Adds a node to the CPPN."""
        eligible_cxs = [
            cx for cx in self.connection_genome if not cx.is_recurrent]
        if len(eligible_cxs) < 1:
            return
        old = np.random.choice(eligible_cxs)
        new_node = Node(choose_random_function(self.config),
                        NodeType.HIDDEN, self.get_new_node_id(), 999)
        self.node_genome.append(new_node)  # add a new node between two nodes
        old.enabled = False  # disable old connection

        # The connection between the first node in the chain and the
        # new node is given a weight of one and the connection between
        # the new node and the last node in the chain
        # is given the same weight as the connection being split
        self.connection_genome.append(Connection(
            self.node_genome[old.from_node.id],
            self.node_genome[new_node.id],
            1.0))

        self.connection_genome.append(Connection(
            self.node_genome[new_node.id],
            self.node_genome[old.to_node.id],
            old.weight))


        self.update_node_layers()

    def remove_node(self):
        """Removes a node from the CPPN."""
        # This is a bit of a buggy mess
        hidden = self.hidden_nodes()
        if len(hidden) < 1:
            return
        node_id_to_remove = np.random.choice([n.id for n in hidden], 1)[0]
        for cx in self.connection_genome[::-1]:
            if node_id_to_remove in [cx.from_node.id, cx.to_node.id]:
                self.connection_genome.remove(cx)
        for node in self.node_genome[::-1]:
            if node.id == node_id_to_remove:
                self.node_genome.remove(node)
                break

        for _, cx in enumerate(self.connection_genome):
            cx.innovation = Connection.get_innovation(
                cx.from_node, cx.to_node)  # definitely wrong
        self.update_node_layers()
        # self.disable_invalid_connections()

    def disable_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections())
        if len(eligible_cxs) < 1:
            return
        cx = np.random.choice(eligible_cxs)
        cx.enabled = False

    def update_node_layers(self) -> int:
        """Update the node layers."""
        layers = feed_forward_layers(self)

        for _, node in enumerate(self.input_nodes()):
            node.layer = 0
        for layer_index, layer in enumerate(layers):
            for _, node_id in enumerate(layer):
                node = find_node_with_id(self.node_genome, node_id)
                node.layer = layer_index + 1

    def input_nodes(self) -> list:
        """Returns a list of all input nodes."""
        return list(filter(lambda n: n.type == NodeType.INPUT, self.node_genome))

    def output_nodes(self) -> list:
        """Returns a list of all output nodes."""
        return list(filter(lambda n: n.type == NodeType.OUTPUT, self.node_genome))

    def hidden_nodes(self) -> list:
        """Returns a list of all hidden nodes."""
        return list(filter(lambda n: n.type == NodeType.HIDDEN, self.node_genome))

    def set_inputs(self, inputs):
        """Sets the input neurons outputs to the input values."""
        if self.config.use_radial_distance:
            # d = sqrt(x^2 + y^2)
            inputs.append(math.sqrt(inputs[0]**2 + inputs[1]**2))
        if self.config.use_input_bias:
            inputs.append(1.0)  # bias = 1.0

        for i, inp in enumerate(inputs):
            # inputs are first N nodes
            self.node_genome[i].sum_input = inp
            self.node_genome[i].output = self.node_genome[i].activation(inp)

    def get_layer(self, layer_index):
        """Returns a list of nodes in the given layer."""
        for node in self.node_genome:
            if node.layer == layer_index:
                yield node

    def get_layers(self):
        """Returns a dictionary of lists of nodes in each layer."""
        layers = {}
        for node in self.node_genome:
            if node.layer not in layers:
                layers[node.layer] = []
            layers[node.layer].append(node)
        return layers

    def clamp_weights(self):
        """Clamps all weights to the range [-max_weight, max_weight]."""
        for cx in self.connection_genome:
            if cx.weight < self.config.weight_threshold and cx.weight >\
                 -self.config.weight_threshold:
                cx.weight = 0
            if cx.weight > self.config.max_weight:
                cx.weight = self.config.max_weight
            if cx.weight < -self.config.max_weight:
                cx.weight = -self.config.max_weight

    def reset_activations(self):
        """Resets all node activations to zero."""
        for node in self.node_genome:
            node.sum_inputs = np.ones((self.config.res_h, self.config.res_w), dtype= CPPN.pixel_inputs.dtype)
            node.outputs = np.ones((self.config.res_h, self.config.res_w), dtype= CPPN.pixel_inputs.dtype)

    def eval(self, inputs):
        """Evaluates the CPPN."""
        self.set_inputs(inputs)
        return self.feed_forward()

    def feed_forward(self):
        """Feeds forward the network."""
        if self.config.allow_recurrent:
            for i in range(self.n_inputs):
                # input nodes (handle recurrent)
                for node_input in list(filter(lambda x,
                    index=i: x.to_node.id == self.node_genome[index].id,
                    self.enabled_connections())):
                    self.node_genome[i].sum_input += node_input.from_node.outputs * node_input.weight

                self.node_genome[i].outputs =\
                    self.node_genome[i].activation(self.node_genome[i].sum_input)

        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer

        for layer_index in range(1, output_layer+1):
            # hidden and output layers:
            layer = self.get_layer(layer_index)
            for node in layer:
                node.sum_input = 0
                node.outputs = 0
                node_inputs = list(
                    filter(lambda x, n=node: x.to_node.id == n.id,
                        self.enabled_connections()))  # cxs that end here
                for cx in node_inputs:
                    node.sum_input += cx.from_node.outputs * cx.weight

                node.output = node.activation(node.sum_input)  # apply activation
                # node.output = np.clip(node.output, -1, 1) # clip output

        return [node.output for node in self.output_nodes()]

    def get_image(self, force_recalculate=False):
        """Returns an image of the network."""
        # decide if we need to recalculate the image
        recalculate = False
        recalculate = recalculate or force_recalculate
        if isinstance(self.image, np.ndarray):
            recalculate = recalculate or self.config.res_h == self.image.shape[0]
            recalculate = recalculate or self.config.res_w == self.image.shape[1]
        else:
            # no cached image
            recalculate = True
        if not recalculate:
            # return the cached image
            return self.image

        if self.config.allow_recurrent:
            # pixel by pixel (good for debugging/recurrent)
            self.image = self.get_image_data_serial()
        else:
            # whole image at once (100x faster)
            self.image = self.get_image_data_parallel()
        return self.image

    def get_image_data_serial(self):
        """Evaluate the network to get image data by processing each pixel
        serially. Much slower than the parallel method, but required if the
        network has recurrent connections."""
        res_h, res_w = self.config.res_h, self.config.res_w
        pixels = []
        for x in np.linspace(-.5, .5, res_w):
            for y in np.linspace(-.5, .5, res_h):
                outputs = self.eval([x, y])
                pixels.extend(outputs)
        if len(self.config.color_mode)>2:
            pixels = np.reshape(pixels, (res_w, res_h, self.n_outputs))
        else:
            pixels = np.reshape(pixels, (res_w, res_h))

        self.image = pixels
        return pixels

    def get_image_data_parallel(self):
        """Evaluate the network to get image data in parallel"""
        res_h, res_w = self.config.res_h, self.config.res_w
        if CPPN.pixel_inputs is None or CPPN.pixel_inputs.shape != (res_h,res_w):
            # initialize inputs if the resolution changed
            CPPN.initialize_inputs(res_h, res_w,
                self.config.use_radial_distance,
                self.config.use_input_bias,
                self.n_inputs)

        # reset the activations to 0 before evaluating
        self.reset_activations()

        layers = feed_forward_layers(self) # get layers
        layers.insert(0, [n.id for n in self.input_nodes()]) # add input nodes as first layer

        for layer in layers:
            # iterate over layers
            for node_index, node_id in enumerate(layer):
                # iterate over nodes in layer
                
                node = find_node_with_id(self.node_genome, node_id) # the current node

                # find incoming connections
                node_inputs = list(
                    filter(lambda x, n=node: x.to_node.id == n.id,
                        self.enabled_connections()))  # cxs that end here

                # initialize the sum_inputs for this node
                node.sum_inputs = np.zeros((res_h, res_w))
                if node.type == NodeType.INPUT:
                    # add pixel inputs to input layer
                    node.sum_inputs += CPPN.pixel_inputs[:,:,node_index]

                for cx in node_inputs:
                    if cx.from_node.outputs is not None:
                        inputs = cx.from_node.outputs * cx.weight
                        node.sum_inputs = node.sum_inputs + inputs

                node.outputs = node.activation(node.sum_inputs)  # apply activation

        outputs = np.array([np.array(node.outputs) for node in self.output_nodes()])
        print(outputs)
        if len(self.config.color_mode)>2:
            outputs =  np.array(outputs).transpose(1, 2, 0) # move color axis to end
        else:
            outputs = np.reshape(outputs, (res_h, res_w))

        self.image = outputs

        self.normalize_image()
        
        return self.image
    
    def normalize_image(self):
        """Normalize from -1 through 1 to 0 through 255 and convert to ints"""
        self.image = np.clip(self.image, -1, 1)
        self.image = 1.0 - np.abs(self.image)
        max_value = np.max(self.image)
        min_value = np.min(self.image)
        image_range = max_value - min_value
        if image_range == 0:
            image_range = 1e-6
        self.image -= min_value
        self.image *= 255
        self.image /= image_range
        self.image = self.image.astype(np.uint8)

    def crossover(self, other_parent):
        """Crossover with another CPPN using the method in Stanley and Miikkulainen (2007)."""
        child = CPPN(self.config) # create child

        # disjoint/excess genes are inherited from first parent
        child.node_genome = copy.deepcopy(self.node_genome)
        child.connection_genome = copy.deepcopy(self.connection_genome)

        # line up by innovation number and find matches
        # child.connection_genome.sort(key=lambda x: x.innovation)
        matching1, matching2 = get_matching_connections(
            self.connection_genome, other_parent.connection_genome)
        print("1", self.connection_genome)
        print()
        print("2", other_parent.connection_genome)
        print()
        print("MATCH1", matching1)
        print()
        print("MATCH2", matching2)
        for match_1, match_2 in zip(matching1, matching2):
            child_cx = child.connection_genome[[x.innovation\
                for x in child.connection_genome].index(
                match_1.innovation)]

            # Matching genes are inherited randomly
            inherit_from_parent_1 = np.random.rand() < .5
            if inherit_from_parent_1:
                child_cx.weight = match_1.weight
                new_from = copy.deepcopy(match_1.from_node)
                new_to = copy.deepcopy(match_1.to_node)
            else:
                child_cx.weight = match_2.weight
                new_from = copy.deepcopy(match_2.from_node)
                new_to = copy.deepcopy(match_2.to_node)

            # assign new nodes and connections
            child_cx.from_node = new_from
            child_cx.to_node = new_to
            existing = find_node_with_id(child.node_genome, new_from.id)
            index_existing = child.node_genome.index(existing)
            child.node_genome[index_existing] = new_from
        
            existing = find_node_with_id(child.node_genome, new_to.id)
            index_existing = child.node_genome.index(existing)
            child.node_genome[index_existing] = new_to

            if(not match_1.enabled or not match_2.enabled):
                if np.random.rand() < 0.75:  # 0.75 from Stanley/Miikulainen 2007
                    child_cx.enabled = False

        # child.update_node_layers()
        print()
        print("CHILD", child.connection_genome)

        return child
