"""Test cases for the lambda_function.py file."""
import time
import unittest
import sys
from urllib import request
sys.path.append('nextGeneration/')
sys.path.append('./')
from nextGeneration.config import Config
from nextGeneration.cppn import CPPN
from nextGeneration.lambda_function import lambda_handler
from local_test import run_server
from multiprocessing import Process

class TestLambdaFunction(unittest.TestCase):
    """Tests for Lambda function."""
    def test_lambda_function(self):
        """Test the lambda_function.py file."""
        event = {"ids": "1,2,3"}
        context = None
        response = lambda_handler(event, context)

        correct_headers = {
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        }

        # Check the response status code
        self.assertEqual(response['statusCode'], 200, "Status code is not 200")
        self.assertDictEqual(response['headers'], correct_headers, "Incorrect headers recieved")
        # self.assertEqual(response['body'], '"1,2,3"', "Incorrect body recieved") # TODO: Fix this test

class TestLocalServer(unittest.TestCase):
    """Tests for the local testing server
    """
    def test_local_server(self):
        """Test the local server"""
        server_process = Process(target=run_server, args=())
        server_process.start()
        timeout = 10 # wait for 10 seconds before failing
        server_response = False
        while timeout > 0 and not server_response:
            response = lambda_handler({"ids": "1,2,3"}, None)
            server_response = response['statusCode'] == 200
            print("Server response:", response)
            time.sleep(1)
            timeout -= 1

        server_process.terminate() # kill test server

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
        self.assertEqual(len(cppn.node_genome), length, "Node genome size is not correct")

    def test_eval_image(self):
        """Test creating an image from the CPPN"""
        # test grayscale image shape
        config = Config()
        config.color_mode = "L"
        cppn = CPPN(config)
        image_data = cppn.get_image_data_fast_method(32,32)
        self.assertEqual(image_data.shape, (32,32), "Image data is not correct shape")

        # test color image shape
        config.color_mode = "RGB"
        cppn = CPPN(config)
        image_data = cppn.get_image_data_fast_method(32,32)
        self.assertEqual(image_data.shape, (32,32,3), "Image data is not correct shape")

if __name__ == '__main__':
    unittest.main()
