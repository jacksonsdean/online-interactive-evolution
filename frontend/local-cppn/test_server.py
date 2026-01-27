from flask import Flask, request
import flask
from flask_restful import Resource, Api
import sys
import os
from flask import Response

app = Flask(__name__,
            static_url_path='', 
            static_folder='.',
            template_folder='web/templates')

api = Api(app)
port = 5100

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def get_resource(path):  # pragma: no cover
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    complete_path = os.path.join(root_dir(), path)
    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    print("CP:", complete_path)
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)

if sys.argv.__len__() > 1:
    port = sys.argv[1]
print("Api running on port : {} ".format(port))

if __name__ == '__main__':
    app.run(port=port,debug=True)