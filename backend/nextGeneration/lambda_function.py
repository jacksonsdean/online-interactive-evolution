"""Handler for Next Generation."""
import json
from config import Config
from cppn import CPPN

HEADERS = {
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        }

def initial_population(config):
    """Create the initial population."""
    population = []
    # create population
    for _ in range(config.population_size):
        population.append(CPPN(config))
    # evaluate population
    for individual in population:
        individual.get_image_data_fast_method(config.res_h, config.res_w)
    population_json = [individual.to_json() for individual in population]
    json_data = json.dumps(population_json)
    return json_data


def lambda_handler(event, context):
    """Handle an incoming request from Next Generation."""
    body = None
    status = 200
    try:
        print("event:", event)
        print("context:", context)
        operation = event['operation']
        config = Config.create_from_json(event['config'])
        if operation == 'reset':
            body = initial_population(config)

    except Exception as e:
        print("ERROR while handling lambda:", type(e), e)
        status = 500
        body = json.dumps(f"error in lambda: {type(e)}: {e}")

    return {
            'statusCode': status,
            'headers': HEADERS,
            'body': body
        }
