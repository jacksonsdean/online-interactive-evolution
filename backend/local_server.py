"""A server for running the backend locally.
Essentially a wrapper for the lambda_handler method,
which is normally called by AWS through a REST API.
"""

from flask import Flask, request
from flask_cors import CORS

from nextGeneration.lambda_function import lambda_handler

class LocalServer(Flask):
    """A simple Flask server for testing the lambda_function.py file locally."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CORS(self) # Allow cross-origin requests

    def run(self, *args, **kwargs):
        print("Local server starting")
        super().run(*args, **kwargs)

server = LocalServer(__name__)
@server.route('/', methods=['POST', 'GET'])
def test():
    """Handle an incoming request from Next Generation."""
    return lambda_handler(request.get_json(), None)

def run_server():
    """Run a local server for testing"""
    server.run(host='0.0.0.0',port=5000)

if __name__ == "__main__":
    run_server()
