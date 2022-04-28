"""Handler for Next Generation."""
import json
import logging
try:
    from config import Config
    from cppn import CPPN
except ModuleNotFoundError:
    from nextGeneration.config import Config
    from nextGeneration.cppn import CPPN

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
    json_data = {"config":config.to_json(), "population": population_json}
    return json_data

def next_generation(config, population_data):
    """Create the next generation."""
    population = []

    # create population
    for individual in population_data:
        population.append(CPPN.create_from_json(individual, config))
    # mutate population
    for individual in population:
        individual.mutate()
    # evaluate population
    for individual in population:
        individual.get_image_data_fast_method(config.res_h, config.res_w)
    population_json = [individual.to_json() for individual in population]
    json_data = {"config":config.to_json(), "population": population_json}
    return json_data


def lambda_handler(event, context):
    """Handle an incoming request from Next Generation."""
    body = None
    status = 200
    try:
        # print("event:", event)
        # print("context:", context)
        data = event['body']
        operation = data['operation']
        config = Config.create_from_json(data['config'])
        if operation == 'reset':
            body = initial_population(config)
        if operation == 'next_gen':
            raw_pop = data['population']

            body = next_generation(config, raw_pop)

    except Exception as e: # pylint: disable=broad-except
        print("ERROR while handling lambda:", type(e), e)
        status = 500
        body = json.dumps(f"error in lambda: {type(e)}: {e}")
        logging.exception(e)

    return {
            'statusCode': status,
            'headers': HEADERS,
            'body': body
        }
