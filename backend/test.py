"""Test cases for the lambda_function.py file."""
import unittest
import time

import numpy as np
from nextGeneration.activation_functions import identity
from nextGeneration.config import Config
from nextGeneration.cppn import CPPN, Connection, Node, NodeType
from nextGeneration.lambda_function import lambda_handler, HEADERS
from local_test import run_server
from multiprocessing import Process
import requests
import json
import sys
sys.path.append('nextGeneration/')
sys.path.append('./')


# TODO DELETE:
import networkx as nx
import matplotlib.pyplot as plt
def visualize_network(individual,sample_point=[.25, .25], color_mode="L", visualize_disabled=False, layout='multi', sample=False, show_weights=False, use_inp_bias=False, use_radial_distance=True):
    if(sample):
        individual.eval(sample_point)
        
    nodes = individual.node_genome
    connections = individual.connection_genome

    max_weight = individual.config.max_weight

    G = nx.DiGraph()
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'darkviolet',
    #         'hotpink', 'chocolate', 'lawngreen', 'lightsteelblue']
    colors = ['lightsteelblue'] * len([node.activation for node in individual.node_genome])
    node_labels = {}

    node_size = 2000
    plt.figure(figsize=(8, 8))
    # plt.figure(figsize=(int(1+(len(individual.get_layers()))*1.5), 6), frameon=False)
    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=0, hspace=0)

    fixed_positions={}
    for i, node in enumerate(individual.input_nodes()):
        G.add_node(node, color=colors[0], shape='d', layer=(node.layer))
        labels = ['y','x','d','b'] if len(individual.input_nodes()) == 4 else ['y','x','b/d']
        node_labels[node] = f"{node.id}\n{labels[i]}:\n{node.activation.__name__}\n"+(f"{node.outputs:.3f}" if node.outputs!=None else "")
        fixed_positions[node] = (-4,((i+1)*2)/len(individual.input_nodes()))
    for node in individual.hidden_nodes():
        G.add_node(node, color=colors[0], shape='o', layer=(node.layer))
        node_labels[node] = f"{node.id}\n{node.activation.__name__}\n"+(f"{node.outputs:.3f}" if node.outputs!=None else "" )

    for i, node in enumerate(individual.output_nodes()):
        title = color_mode[i] if i < len(color_mode) else 'XXX'
        G.add_node(node, color=colors[0], shape='s', layer=(node.layer))
        node_labels[node] = f"{node.id}\n{title}:\n"+(f"{node.outputs:.3f}" if node.outputs!=None else "" )
        fixed_positions[node] = (4, ((i+1)*2)/len(individual.output_nodes()))

    pos = {}
    if(layout=='multi'):
        pos=nx.multipartite_layout(G, scale=4, subset_key='layer')
    elif(layout=='spring'):
        pos=nx.spring_layout(G, scale=4)

    shapes = set((node[1]["shape"] for node in G.nodes(data=True)))
    for shape in shapes:
        nodes = [sNode[0] for sNode in filter(
            lambda x: x[1]["shape"] == shape, G.nodes(data=True))]
        colors = [nx.get_node_attributes(G, 'color')[cNode] for cNode in nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors,
                            label=node_labels, node_shape=shape, nodelist=nodes)

    edge_labels = {}
    for cx in connections:
        if(not visualize_disabled and (not cx.enabled or np.isclose(cx.weight, 0))): continue
        style = ('-', 'k',  .5+abs(cx.weight)/max_weight) if cx.enabled else ('--', 'grey', .5+ abs(cx.weight)/max_weight)
        if(cx.enabled and cx.weight<0): style  = ('-', 'r', .5+abs(cx.weight)/max_weight)

        G.add_edge(cx.from_node, cx.to_node, weight=f"{cx.weight:.4f}", pos=pos, style=style)
        edge_labels[(cx.from_node, cx.to_node)] = f"{cx.weight:.3f}"


    edge_colors = nx.get_edge_attributes(G,'color').values()
    edge_styles = shapes = set((s[2] for s in G.edges(data='style')))
    use_curved = show_weights or len(individual.get_layers())<3
    for s in edge_styles:
        edges = [e for e in filter(
            lambda x: x[2] == s, G.edges(data='style'))]
        nx.draw_networkx_edges(G, pos,
                                edgelist=edges,
                                arrowsize=25, arrows=True, 
                                node_size=[node_size]*1000,
                                style=s[0],
                                edge_color=[s[1]]*1000,
                                width =s[2],
                                connectionstyle= "arc3" if use_curved else "arc3,rad=0.2"
                                # connectionstyle= "arc3"
                            )

    if (show_weights):
        nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=.75)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.tight_layout()
    plt.show()


TEST_CONFIG = {"population_size": 10, "activations": [
    "identity"], "output_activation": "tanh"}

TEST_RESET_EVENT = {
    "operation": "reset", "config": TEST_CONFIG,
}

TEST_POPULATION =\
    [CPPN(Config.create_from_json(TEST_CONFIG)).to_json()
     for _ in range(TEST_CONFIG["population_size"])]

TEST_NEXT_GEN_EVENT = {
    "operation": "next_gen", "config": TEST_CONFIG, "population": TEST_POPULATION
}

TEST_POST_REQUEST_FORMAT = {
    'resource': '/next-gen', 'path': '/next-gen', 'httpMethod': 'POST',
    'headers': None, 'multiValueHeaders': None, 'queryStringParameters': None,
    'multiValueQueryStringParameters': None, 'pathParameters': None, 'stageVariables': None,
    'requestContext': None, 'body': None, 'isBase64Encoded': False
}


class TestLambdaFunction(unittest.TestCase):
    """Tests for Lambda function."""

    def test_initial_population(self):
        """Test the creation of an initial population in the lambda function."""
        post_json = TEST_POST_REQUEST_FORMAT
        post_json['body'] = TEST_RESET_EVENT
        response = lambda_handler(post_json, None)
        self.assertEqual(response['statusCode'], 200, "Status code is not 200")
        self.assertDictEqual(response['headers'],
                             HEADERS, "Incorrect headers received")
        response = json.loads(response['body'])
        response_genomes = response["population"]
        self.assertEqual(len(response_genomes),
                         TEST_CONFIG['population_size'],
                         "Incorrect number of CPPNs in response")

    def test_next_gen(self):
        """Test moving to the next generation in the lambda function."""
        print("Testing next generation...")
        post_json = TEST_POST_REQUEST_FORMAT
        event = TEST_NEXT_GEN_EVENT
        event["population"][0]["selected"] = True
        event["population"][3]["selected"] = True
        post_json['body'] = event
        response = lambda_handler(post_json, None)
        self.assertEqual(response['statusCode'], 200, "Status code is not 200")
        self.assertDictEqual(response['headers'],
                             HEADERS, "Incorrect headers received")
        response = json.loads(response['body'])
        response_genomes = response["population"]
        self.assertEqual(len(response_genomes),
                         TEST_CONFIG['population_size'],
                         "Incorrect number of CPPNs in response")

        self.assertTrue(
            response_genomes[0]["selected"], "First CPPN is not selected")


class TestLocalServer(unittest.TestCase):
    """Tests for the local testing server
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_process = None
        self.addCleanup(self.cleanup)

    def cleanup(self):
        """Cleanup the server process"""
        if self.server_process is not None and self.server_process.is_alive():
            self.server_process.terminate()

    def test_local_server(self):
        """Test the local server"""
        self.server_process = Process(target=run_server, args=())
        self.server_process.start()
        timeout = 10  # wait for 10 seconds before failing
        server_response = False
        post_json = TEST_POST_REQUEST_FORMAT
        post_json['body'] = TEST_RESET_EVENT
        while timeout > 0 and not server_response:
            response = requests.post(
                "http://localhost:5000", json=post_json, headers=HEADERS)
            server_response = response.status_code == 200
            time.sleep(1)
            timeout -= 1

        self.server_process.terminate()  # kill test server

        self.assertTrue(server_response,
                        "Local server did not respond with correct response in time")


class TestCPPN(unittest.TestCase):
    """Test the CPPN class"""

    def test_create_cppn(self):
        """Test the CPPN creation process"""
        config = Config()
        cppn = CPPN(config)
        self.assertIsInstance(cppn, CPPN, "CPPN is not an instance of CPPN")
        length = cppn.n_inputs + cppn.n_outputs + config.hidden_nodes_at_start
        self.assertEqual(len(cppn.node_genome), length,
                         "Node genome size is not correct")

    def test_eval_image(self):
        """Test creating an image from the CPPN"""
        # test grayscale image shape
        config = Config()
        config.color_mode = "L"
        cppn = CPPN(config)
        image_data = cppn.get_image_data_fast_method(32, 32)
        self.assertEqual(image_data.shape, (32, 32),
                         "Image data is not correct shape")

        # test color image shape
        config.color_mode = "RGB"
        cppn = CPPN(config)
        image_data = cppn.get_image_data_fast_method(32, 32)
        self.assertEqual(image_data.shape, (32, 32, 3),
                         "Image data is not correct shape")

    def test_update_node_layers(self):
        """Test updating the node layers"""
        config = Config()
        config.hidden_nodes_at_start = 2
        cppn = CPPN(config)
        self.assertEqual(len(cppn.get_layers()), 3,
                         "Node layers are not correct length after initializing")

        # add a node to increase the number of layers
        connection_to_split = cppn.connection_genome[-1]
        new_node = Node(identity, NodeType.HIDDEN, len(cppn.node_genome))
        cppn.node_genome.append(new_node)
        connection_to_split.enabled = False  # disable old connection
        cppn.connection_genome.append(Connection(
            cppn.node_genome[connection_to_split.from_node.id],
            cppn.node_genome[new_node.id],
            1))

        cppn.connection_genome.append(Connection(
            cppn.node_genome[new_node.id], 
            cppn.node_genome[connection_to_split.to_node.id],
            connection_to_split.weight))

        cppn.update_node_layers()
        self.assertEqual(len(cppn.get_layers()), 4,
                         "Node layers are not correct length after adding new node")

    def test_serialization(self):
        """Serialize and deserialize a CPPN, ensuring that the content remains the same"""
        config = Config()
        config.color_mode = "RGB"
        cppn = CPPN(config)
        serialized_cppn = cppn.to_json()
        cppn_from_json = CPPN.create_from_json(serialized_cppn, config)
        self.assertEqual(len(cppn.node_genome), len(cppn_from_json.node_genome),
                         "Node genome has wrong number of elements after serialization")
        self.assertEqual(len(cppn.connection_genome), len(cppn_from_json.connection_genome),
                         "Connection genome has wrong number of elements after serialization")

        for node1, node2 in zip(cppn.node_genome, cppn_from_json.node_genome):
            self.assertEqual(node1.id, node2.id,
                             "Node genome has node with wrong id after serialization")
            self.assertEqual(node1.activation, node2.activation,
                             "Node genome has node with wrong activation after serialization")
            self.assertEqual(node1.layer, node2.layer,
                             "Node genome has node with wrong layer after serialization")

        for connection1, connection2 in zip(cppn.connection_genome, cppn_from_json.connection_genome):
            self.assertEqual(connection1.innovation, connection2.innovation,
                             "Connection genome has connection with wrong innovation number after serialization")
            self.assertEqual(connection1.enabled, connection2.enabled,
                             "Connection genome has connection with wrong enabled after serialization")
            self.assertEqual(connection1.is_recurrent, connection2.is_recurrent,
                             "Connection genome has connection with wrong is_recurrent after serialization")
            self.assertEqual(connection1.weight, connection2.weight,
                             "Connection genome has connection with wrong weight after serialization")
            self.assertEqual(connection1.from_node.id, connection2.from_node.id,
                             "Connection genome has connection with wrong input node after serialization")
            self.assertEqual(connection1.to_node.id, connection2.to_node.id,
                             "Connection genome has connection with wrong output node after serialization")

        img1 = cppn.get_image_data_fast_method(32, 32)
        img2 = cppn_from_json.get_image_data_fast_method(32, 32)
        self.assertEqual(img1.shape, img2.shape,
                         "Image data has wrong shape after serialization")
        self.assertEqual(img1.dtype, img2.dtype,
                         "Image data has wrong dtype after serialization")
        import matplotlib.pyplot as plt
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        self.assertTrue(np.all(img1 == img2),
                        "Image data has wrong data after serialization")


if __name__ == '__main__':
    unittest.main()
