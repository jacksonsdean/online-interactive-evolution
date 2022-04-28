export const API_URL = process.env.REACT_APP_NODE_ENV === "local" ? "http://127.1.1.1:5000" : (process.env.REACT_APP_NODE_ENV === "production" ? "https://5c9gzxp0r0.execute-api.us-east-1.amazonaws.com/prod/next-gen" : "https://vln9z6a7gi.execute-api.us-east-1.amazonaws.com/development/next-gen")

export const INITIAL_REQUEST = {
    "operation": "reset", "config": { "color_mode": "RGB", "res_w":64,"res_h":64, "population_size": 18, "activations": ["identity", "gauss", "sin", "cos", "sawtooth", "tanh", "sigmoid"], "output_activation": "tanh" },
}