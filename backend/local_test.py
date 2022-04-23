from nextGeneration.lambda_function import lambda_handler
from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def test():
    """Handle an incoming request from Next Generation."""
    print(request.data)
    return lambda_handler({"queryStringParameters": {"ids": "1,2,3"}}, None)

if __name__ == "__main__":
    app.run()