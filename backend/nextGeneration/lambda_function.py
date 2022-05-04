"""Handler for Next Generation."""
import base64
import copy
import io
import json
import logging
import random
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
    """Evaluate the population to generate images and return the json data."""
    for individual in population:
        if individual.selected:
            continue # frontend already has image data for previous gen
        # evaluate the CPPN
        individual.get_image()
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

def selection_procedure(config, population, selected):
    """Creates a new population given the selected individuals in-place

    Args:
        config (Config): The configuration for evolution
        population (list[CPPN]): The current population
        selected (list[CPPN]): The selected individuals
    """
    for index, _ in enumerate(population):
        if not population[index].selected:
            do_crossover = config.do_crossover and np.random.random() < config.prob_crossover
            if do_crossover:
                # crossover two selected individuals
                parent1 = np.random.choice(selected)
                parent2 = np.random.choice(selected)
                population[index] = parent1.crossover(parent2)
            else:
                # replace with a mutated version of a random selected individual
                random_parent = np.random.choice(selected)
                population[index] = copy.deepcopy(random_parent) # make a copy

            # mutate
            population[index].mutate()
            population[index].selected = False # deselect

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

    # replace the unselected individuals with new individuals
    selection_procedure(config, population, selected)

    # evaluate population
    json_data = evaluate_population(population, config)
    return json_data

def save_images(config, population_data):
    """Return the images in the population, generated with save resolution."""# create population
    population = []
    for individual in population_data:
        population.append(CPPN.create_from_json(individual, config))

    for individual in population:
        if individual.selected:
            # apply save resolution before evaluating
            individual.config.res_h = config.save_h
            individual.config.res_w = config.save_w

        individual.selected = not individual.selected # invert selection

    json_data = evaluate_population(population, config)
    return json_data

def lambda_handler(event, context):
    """Handle an incoming request from Next Generation."""
    # pylint: disable=unused-argument #(context required by lambda)

    body = None
    status = 200
    try:
        data = event['body'] if 'body' in event else event
        if isinstance(data, str):
            # load the data to a json object
            data = json.loads(data, strict=False)
        operation = data['operation']

        config = Config.create_from_json(data['config'])

        # use the seed from the config
        random.seed(int(config.seed))
        np.random.seed(int(config.seed))

        if operation == 'reset':
            body = initial_population(config)
        elif operation == 'next_gen':
            raw_pop = data['population']
            body = next_generation(config, raw_pop)
        elif operation == 'save_images':
            raw_pop = data['population']
            body = save_images(config, raw_pop)

    except Exception as exception: # pylint: disable=broad-except
        print("ERROR while handling lambda:", type(exception), exception)
        status = 500
        body = json.dumps(f"error in lambda: {type(exception)}: {exception}")
        logging.exception(exception) # okay to disable broad-except

    return {
            'statusCode': status,
            'headers': HEADERS,
            'body': body
        }
