from nextGeneration.lambda_function import lambda_handler
from flask import Flask, request
from flask_cors import CORS


class LocalServer(Flask):
    """A simple Flask server for testing the lambda_function.py file locally."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CORS(self) # Allow cross-origin requests

    def run(self, *args, **kwargs):
        print("Local server starting")
        super().run(*args, **kwargs)


def run_server():
    """Create and run a local server for testing"""
    server = LocalServer(__name__)
    server.run()


    @server.route('/', methods=['POST', 'GET'])
    def test():
        """Handle an incoming request from Next Generation."""
        print("request:", request)
        return lambda_handler(request.get_json(), None)


if __name__ == "__main__":
    run_server()
