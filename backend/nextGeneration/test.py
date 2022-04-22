"""Test cases for the lambda_function.py file."""
import unittest
from config import Config
from cppn import CPPN
from lambda_function import lambda_handler

class TestLambdaFunction(unittest.TestCase):
    """Tests for Lambda function."""
    def test_lambda_function(self):
        """Test the lambda_function.py file."""
        event = {"queryStringParameters": {"ids": "1,2,3"}}
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
        self.assertEqual(response['body'], '"1,2,3"', "Incorrect body recieved")

class TestCPPN(unittest.TestCase):
    """Test the CPPN class"""
    def test_create_cppn(self):
        """Test the CPPN creation process"""
        config = Config()
        cppn = CPPN(config)
        self.assertIsInstance(cppn, CPPN, "CPPN is not an instance of CPPN")
        length = config.num_inputs + config.num_outputs + config.hidden_nodes_at_start
        self.assertEqual(len(cppn.node_genome), length, "Node genome size is not correct")

    def test_eval_image(self):
        """Test creating an image from the CPPN"""
        config = Config()
        cppn = CPPN(config)
        image_data = cppn.get_image_data_fast_method(32,32,"L")
        self.assertEqual(image_data.shape, (32,32), "Image data is not correct shape")
        image_data = cppn.get_image_data_fast_method(32,32,"RGB")
        self.assertEqual(image_data.shape, (32,32,3), "Image data is not correct shape")

if __name__ == '__main__':
    unittest.main()
