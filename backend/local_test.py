from nextGeneration.lambda_function import lambda_handler
from flask import Flask

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
    return lambda_handler({"queryStringParameters": {"ids": "1,2,3"}}, None)
