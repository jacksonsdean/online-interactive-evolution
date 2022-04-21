"""Test cases for the lambda_function.py file."""
import unittest
from nextGeneration.lambda_function import lambda_handler

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

if __name__ == '__main__':
    unittest.main()
