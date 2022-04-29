import { NEXT_GEN_OPERATION, POST_FORMAT, RESET_OPERATION } from 'Constants';
import { API_URL } from 'Constants';

export function post(url, data) {
    return fetch(url, { method: "POST", headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}

export function nextGeneration(currentPopulation, config) {
    let at_least_one_selected = false;
    for (let i = 0; i < currentPopulation.length; i++) {
        if (currentPopulation[i].selected) {
            at_least_one_selected = true;
            break;
        }
    }
    if (!at_least_one_selected) {
        return Promise.reject("No individuals selected");
    }
    const postData = POST_FORMAT
    postData.operation = NEXT_GEN_OPERATION
    let population = JSON.parse(JSON.stringify(currentPopulation)); // clone
    // clear out images
    for (let i = 0; i < population.length; i++) {
        population[i].image = 'undefined'
    }
    console.log(currentPopulation)
    postData.population = population
    postData.config = config
    return new Promise((resolve, reject) => {
        post(API_URL, postData).then((response) => {
            console.log(response)
            if (response.status === 200) {
                resolve(response.json())
            }
            else {
                reject(response.status);
            }
        }
        ).catch((error) => {
            console.log("Error: " + error);
            reject(error);
        })
    })
}

export function initialPopulation(config) {
    let postData = POST_FORMAT
    postData.operation = RESET_OPERATION
    postData.config = config
    return new Promise((resolve, reject) => {
        post(API_URL, postData).then((response) => {
            console.log(response)
            if (response.status === 200) {
                resolve(response.json())
            }
            else {
                reject(response.status);
            }
        }
        ).catch((error) => {
            console.log("Error: " + error);
            reject(error);
        })
    })
}