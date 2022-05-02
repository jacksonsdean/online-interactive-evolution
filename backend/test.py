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
    """Tests for the local testing server"""

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
        # Test shapes:
        config = Config()
        config.color_mode = "L" # test grayscale image shape
        config.res_h, config.res_w = 32, 32
        cppn = CPPN(config)
        image_data = cppn.get_image()
        self.assertEqual(image_data.shape, (32, 32),
                         "Image data is not correct shape")
        self.assertEqual(image_data.dtype, np.uint8,
                            "Image data is not correct dtype")


        config.color_mode = "RGB" # test color image shape
        cppn = CPPN(config)
        image_data = cppn.get_image()
        self.assertEqual(image_data.shape, (32, 32, 3),
                         "Image data is not correct shape")
        self.assertEqual(image_data.dtype, np.uint8,
                            "Image data is not correct dtype")

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

        for cx1, cx2 in zip(cppn.connection_genome, cppn_from_json.connection_genome):
            self.assertEqual(cx1.innovation, cx2.innovation,
                             "Connection genome has connection with wrong innovation number")
            self.assertEqual(cx1.enabled, cx2.enabled,
                             "Connection genome has connection with wrong enabled")
            self.assertEqual(cx1.is_recurrent, cx2.is_recurrent,
                             "Connection genome has connection with wrong is_recurrent")
            self.assertEqual(cx1.weight, cx2.weight,
                             "Connection genome has connection with wrong weight")
            self.assertEqual(cx1.from_node.id, cx2.from_node.id,
                             "Connection genome has connection with wrong input node")
            self.assertEqual(cx1.to_node.id, cx2.to_node.id,
                             "Connection genome has connection with wrong output node")

        img1 = cppn.get_image()
        img2 = cppn_from_json.get_image()
        self.assertEqual(img1.shape, img2.shape,
                         "Image data has wrong shape after serialization")
        self.assertEqual(img1.dtype, img2.dtype,
                         "Image data has wrong dtype after serialization")

        self.assertTrue(np.all(img1 == img2),
                        "Image data has wrong data after serialization")


if __name__ == '__main__':
    unittest.main()
