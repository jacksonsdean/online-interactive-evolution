"""Stores configuration parameters for the CPPN."""
import json

try:
    from activation_functions import get_all #,tanh,identity
    from graph_util import name_to_fn
except ModuleNotFoundError:
    from nextGeneration.activation_functions import get_all #,tanh,identity
    from nextGeneration.graph_util import name_to_fn

class   Config:
    """Stores configuration parameters for the CPPN."""
    def __init__(self) -> None:
        self.population_size = 10
        self.res_w = 28
        self.res_h = 28
        self.color_mode = "RGB"
        self.do_crossover = True
        self.allow_recurrent = False
        self.max_weight = 3.0
        self.weight_threshold = 0
        self.weight_mutation_max = 2
        self.prob_random_restart =.001
        self.prob_weight_reinit = 0.0
        self.prob_reenable_connection = 0.1
        self.init_connection_probability = 1
        self.activations = get_all()
        self.seed = None

        """DGNA: the probability of adding a node is 0.5 and the
        probability of adding a connection is 0.4.
        SGNA: probability of adding a node is 0.05 and the
         probability of adding a connection is 0.04."""
        self.prob_mutate_activation = .1
        self.prob_mutate_weight = .35
        self.prob_add_connection = .4
        self.prob_add_node = .4
        self.prob_remove_node = 0.1
        self.prob_disable_connection = .25

        self.output_activation = None

        # DGNA/SGMA uses 1 or 2 so that patterns in the initial
        # generation would be nontrivial (Stanley, 2007).
        self.hidden_nodes_at_start=1

        self.allow_input_activation_mutation = False

        self.animate = False

        # https://link.springer.com/content/pdf/10.1007/s10710-007-9028-8.pdf page 148
        self.use_input_bias = True # SNGA,
        self.use_radial_distance = True # bias towards radial symmetry


    def fns_to_strings(self):
        """Converts the activation functions to strings."""
        self.activations= [fn.__name__ for fn in self.activations]
        self.output_activation = self.output_activation.__name__ if\
            self.output_activation is not None else ""

    def strings_to_fns(self):
        """Converts the activation functions to functions."""
        self.activations= [name_to_fn(name) for name in self.activations]
        # self.activations.append(avg_pixel_distance_fitness)
        self.output_activation = name_to_fn(self.output_activation)

    def to_json(self):
        """Converts the configuration to a json string."""
        self.fns_to_strings()
        json_string = json.dumps(self.__dict__, sort_keys=True, indent=4)
        self.strings_to_fns()
        return json_string


    def from_json(self, json_dict):
        """Converts the configuration from a json string."""
        if isinstance(json_dict, dict):
            json_dict = json.loads(json_dict)
        self.fns_to_strings()
        for key, value in json_dict.items():
            setattr(self, key, value)
        self.strings_to_fns()

    @staticmethod
    def create_from_json(json_str):
        """Creates a configuration from a json string."""
        if isinstance(json_str, str):
            json_str = json.loads(json_str)
        config = Config()
        for key, value in json_str.items():
            setattr(config, key, value)
        config.strings_to_fns()
        return config
