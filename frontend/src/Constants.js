export const API_URL = process.env.REACT_APP_NODE_ENV === "local" ? "http://127.1.1.1:5000" : (process.env.REACT_APP_NODE_ENV === "production" ? "https://5c9gzxp0r0.execute-api.us-east-1.amazonaws.com/prod/next-gen" : "https://vln9z6a7gi.execute-api.us-east-1.amazonaws.com/development/next-gen")

export const DEFAULT_CONFIG = { 
    "color_mode": "RGB",
    "res_w": 128,
    "res_h": 128,
    "population_size": 9,
    "activations": ["identity", "gauss", "sin", "cos", "sawtooth", "tanh", "sigmoid"],
    "output_activation": "" }

export const RESET_OPERATION = "reset"
export const NEXT_GEN_OPERATION = "next_gen"

export const POST_FORMAT = {
    "operation": RESET_OPERATION, "config": DEFAULT_CONFIG
}

export const MAX_HISTORY = 20