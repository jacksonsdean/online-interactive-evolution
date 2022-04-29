"""Handler for Next Generation."""
import base64
import copy
import io
import json
import logging
from PIL import Image
import numpy as np

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

def evaluate_population(population, config)->str:
    for individual in population:
        # evaluate the CPPN
        individual.get_image_data_fast_method(config.res_h, config.res_w)

        # normalize 0-255 and convert to ints
        individual.image = individual.image / np.max(individual.image)
        individual.image = individual.image * 255
        individual.image = individual.image.astype(np.uint8)

        # convert from numpy to bytes
        individual.image = Image.fromarray(individual.image)
        # convert to RGB if not RGB
        if individual.image.mode != 'RGB':
            individual.image = individual.image.convert('RGB')
        with io.BytesIO() as img_byte_arr:
            individual.image.save(img_byte_arr, format='PNG')
            im_b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf8")
            individual.image = im_b64
            img_byte_arr.close()

    # convert to json
    population_json = [individual.to_json() for individual in population]
    json_data = {"config":config.to_json(), "population": population_json}
    json_data = json.dumps(json_data)
    return json_data

def initial_population(config):
    """Create the initial population."""
    population = []
    # create population
    for _ in range(config.population_size):
        population.append(CPPN(config))
    # evaluate population
    json_data = evaluate_population(population, config)

    return json_data

def next_generation(config, population_data):
    """Create the next generation."""
    population = []
    if population_data is None or len(population_data) == 0:
        # return an initial population
        return initial_population(config)
    # create population
    for individual in population_data:
        population.append(CPPN.create_from_json(individual, config))
    # build list of selected individuals
    selected = list(filter(lambda x: x.selected, population))
    # mutate selected
    for index, _ in enumerate(population):
        if not population[index].selected:
            # replace with a mutated version of a random selected individual
            random_parent = np.random.choice(selected)
            population[index] = copy.deepcopy(random_parent) # make a copy
            population[index].mutate()
            population[index].selected = False # deselect

    # sort by selection status
    population.sort(key=lambda x: x.selected, reverse=True)

    # evaluate population
    json_data = evaluate_population(population, config)
    return json_data


def lambda_handler(event, context):
    """Handle an incoming request from Next Generation."""
    body = None
    status = 200
    try:
        print("event:", event)
        print("context:", context)
        data = event['body'] if 'body' in event else event
        if isinstance(data, str):
            # load the data to a json object
            data = json.loads(data, strict=False)
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
