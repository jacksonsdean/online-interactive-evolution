//import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';
import { computeSimilarities } from './clip.js';

export async function backprop(prompt, cppns, res){
    const optimizer = tf.train.sgd(0.1 /* learningRate */);
    for (let step = 0; step < 10; step++) {
        const sim = await computeSimilarities(prompt, cppns, res);
        console.log("BP Similarities: ", sim);

        // Update weights based on similarities using tensorflow gradient descent to maximize the mean similarity
        optimizer.minimize(() => {
            return tf.tensor1d(sim).mean().neg();
        });

        // Update the CPPN weights
        cppns.forEach(cppn => {
            cppn.connections.forEach(connection => {
                connection.weight = connection.weight - optimizer.learningRate * connection.weight.grad;
            });
        });
    }
}
