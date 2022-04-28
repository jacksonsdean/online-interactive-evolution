import { NEXT_GEN_OPERATION, POST_FORMAT, RESET_OPERATION } from 'Constants';
import { API_URL } from 'Constants';

export function post(url, data) {
    return fetch(url, { method: "POST", headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}

export function nextGeneration(currentPopulation, config) {
    const postData = POST_FORMAT
    postData.operation = NEXT_GEN_OPERATION
    postData.population = JSON.stringify(currentPopulation)
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