// import { pipeline, CLIPTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';
// import { env, AutoTokenizer, CLIPTextModelWithProjection } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';
//import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';
import { pipeline, AutoTokenizer, CLIPTextModelWithProjection, AutoProcessor, CLIPVisionModelWithProjection, RawImage, cos_sim } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

import { env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';
env.allowLocalModels = false; // Skip local model check


import { renderCPPNImages, generateGrid } from './renderer.js';
import { evaluate } from './cppn.js';






















export async function computeSimilarities(prompt, cppns, res){
    // const imageData = renderCPPNImages(cppns, res);  // TODO may not want to render images to canvas at this step

     // Generate grid tensors
    const { X, Y, R } = generateGrid(10, 10, res);
    
    // Create input tensors for the CPPN (stacked inputs)
    const inputTensors = [X.flatten(), Y.flatten(), R.flatten()];
    
    // Evaluate the CPPN to get outputs
    // for loop through cppns
    let image_x = 5;
    let image_y = 5;
    let image_width = res;
    let imageDatas = [];
    for (let i = 0; i < cppns.length; i++) {
        const cppn = cppns[i];
        const outputs = evaluate(cppn, inputTensors);
    
        // Collect RGB channels from the outputs
        const rChannel = outputs[0].reshape([res, res]);
        const gChannel = outputs[1].reshape([res, res]);
        const bChannel = outputs[2].reshape([res, res]);
    
        // Stack RGB channels and normalize for rendering
        const imageDataArray = tf.stack([rChannel, gChannel, bChannel], -1)

        imageDatas.push(imageDataArray);
    }
    imageDatas = tf.stack(imageDatas);
    const sim = await computeSimilaritiesFromOutputs(prompt, imageDatas);

    return sim;

}


export async function computeSimilaritiesFromOutputs(prompt, images){
    console.log("Computing similarities for prompt: ", prompt);
    // Load tokenizer and text model
    const tokenizer = await AutoTokenizer.from_pretrained('jinaai/jina-clip-v1');

    const text_model = await CLIPTextModelWithProjection.from_pretrained('jinaai/jina-clip-v1');
    console.log(text_model)

    // Load processor and vision model
    const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch32');
    const vision_model = await CLIPVisionModelWithProjection.from_pretrained('jinaai/jina-clip-v1');
    
    // Run tokenization
    const texts = [prompt];
    // repeat the prompt for each image
    for (let i = 0; i < images.length - 1; i++) {
        texts.push(prompt);
    }
    const text_inputs = tokenizer(texts, { padding: true, truncation: true });
    
    // Compute text embeddings
    const { text_embeds } = await text_model(text_inputs);
    
    
    const raw_images = images.map(image => new RawImage(image, 256, 256, 3));
    const image_inputs = await processor(raw_images);
    // const image_inputs = await processor(images);
    
    // Compute vision embeddings
    const { image_embeds } = await vision_model(image_inputs);

    //  Compute similarities
    let result = [];
    for (let i = 0; i < images.length; i++) {
        console.log(text_embeds[i].data);
        let sim = cos_sim(text_embeds[i].data, image_embeds[i].data);
        result.push(sim);
    }
    // assert the length of the sim is equal to the number of images
    console.assert(result.length === images.length, "The length of the similarity array is not equal to the number of images");
    // let result = [cos_sim(text_embeds[0].data, image_embeds[0].data)];
    return result;
}



export async function computeSimilaritiesFromRawImagesCPU(prompt, images){
    console.log("Computing similarities for prompt: ", prompt);
    // Load tokenizer and text model
    const tokenizer = await AutoTokenizer.from_pretrained('jinaai/jina-clip-v1');

    const text_model = await CLIPTextModelWithProjection.from_pretrained('jinaai/jina-clip-v1');
    console.log(text_model)

    // Load processor and vision model
    const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch32');
    const vision_model = await CLIPVisionModelWithProjection.from_pretrained('jinaai/jina-clip-v1');
    
    // Run tokenization
    const texts = [prompt];
    // repeat the prompt for each image
    for (let i = 0; i < images.length - 1; i++) {
        texts.push(prompt);
    }
    const text_inputs = tokenizer(texts, { padding: true, truncation: true });
    
    // Compute text embeddings
    const { text_embeds } = await text_model(text_inputs);

    const raw_images = images.map(image => new RawImage(image, 256, 256, 3));
    const image_inputs = await processor(raw_images);
    
    // Compute vision embeddings
    const { image_embeds } = await vision_model(image_inputs);

    //  Compute similarities
    let result = [];
    for (let i = 0; i < images.length; i++) {
        console.log(text_embeds[i].data);
        let sim = cos_sim(text_embeds[i].data, image_embeds[i].data);
        result.push(sim);
    }
    // assert the length of the sim is equal to the number of images
    console.assert(result.length === images.length, "The length of the similarity array is not equal to the number of images");
    // let result = [cos_sim(text_embeds[0].data, image_embeds[0].data)];
    return result;
}




export async function computeSimilaritiesFromRawImagesPipeline(prompt, images){
    console.log("Computing similarities for prompt: ", prompt);
    // Load tokenizer and text model
    const text_model = await pipeline(
        "feature-extraction",
        'Xenova/clip-vit-base-patch16',
        { device: "webgpu" },
      );
    const image_model = await pipeline(
        "image-feature-extraction",
        'Xenova/clip-vit-base-patch16',
        { device: "webgpu" },
      );
    
    // Run tokenization
    const texts = [prompt];
    // repeat the prompt for each image
    for (let i = 0; i < images.length - 1; i++) {
        texts.push(prompt);
    }
    
    // Compute text embeddings
    const { text_embeds } = await text_model(text_inputs);
    
    // Compute vision embeddings
    const { image_embeds } = await vision_model(images);

    //  Compute similarities
    let result = [];
    for (let i = 0; i < images.length; i++) {
        console.log(text_embeds[i].data);
        let sim = cos_sim(text_embeds[i].data, image_embeds[i].data);
        result.push(sim);
    }
    // assert the length of the sim is equal to the number of images
    console.assert(result.length === images.length, "The length of the similarity array is not equal to the number of images");
    // let result = [cos_sim(text_embeds[0].data, image_embeds[0].data)];
    return result;
}
