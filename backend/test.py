"""Test cases for the lambda_function.py file."""
import unittest
import time
from nextGeneration.config import Config
from nextGeneration.cppn import CPPN
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

        for genome in response_genomes:
            print(genome["selected"])
            print()


        self.assertTrue(response_genomes[0]["selected"], "First CPPN is not selected")


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


if __name__ == '__main__':
    unittest.main()
