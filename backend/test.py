"""Test cases for the lambda_function.py file."""
import sys
sys.path.append('nextGeneration/')
sys.path.append('./')
from multiprocessing import Process
from local_test import run_server
from nextGeneration.lambda_function import lambda_handler, HEADERS
from nextGeneration.cppn import CPPN
from nextGeneration.config import Config
import time
import unittest


TEST_EVENT = {
    "operation": "reset", "config": {"population_size": 10, "activations": ["identity"], "output_activation":"tanh"},
}


class TestLambdaFunction(unittest.TestCase):
    """Tests for Lambda function."""

    def test_lambda_function(self):
        """Test the lambda_function."""
        ...
        # event = {"ids": "1,2,3"}
        # context = None
        # response = lambda_handler(event, context)

        # Check the response status code
        # self.assertEqual(response['statusCode'], 200, "Status code is not 200")
        # self.assertDictEqual(response['headers'], HEADERS, "Incorrect headers received")
        # self.assertEqual(response['body'], '"1,2,3"', "Incorrect body received") # TODO: Fix this test

    def test_initial_population(self):
        """Test the creation of an initial population."""
        event = TEST_EVENT
        response = lambda_handler(event, None)
        self.assertEqual(response['statusCode'], 200, "Status code is not 200")
        self.assertDictEqual(response['headers'],
                             HEADERS, "Incorrect headers received")


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
        event = TEST_EVENT
        while timeout > 0 and not server_response:
            response = lambda_handler(event, None)
            server_response = response['statusCode'] == 200
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
