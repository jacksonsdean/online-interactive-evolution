import re
from nextGeneration.lambda_function import lambda_handler
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Allow cross-origin requests

@app.route('/', methods=['POST', 'GET'])
def test():
    """Handle an incoming request from Next Generation."""
    print("request:", request)
    return lambda_handler(request.get_json(), None)

if __name__ == "__main__":
    app.run()