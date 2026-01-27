import { evaluate } from './cppn.js';
import { initializeCPPN } from './cppn.js';

export function setupGridRenderer(gridResolutionInput, gridResolutionValue, evaluateBtn, cppns) {
  const canvas = document.getElementById('imageCanvas');
  //canvas background black
    // canvas.style.backgroundColor = 'black';
  const ctx = canvas.getContext('2d');
  let resolution = parseInt(gridResolutionInput.value);

  gridResolutionInput.addEventListener('input', () => {
    resolution = parseInt(gridResolutionInput.value);
    gridResolutionValue.textContent = resolution;
  });


    // Evaluate button listener
    evaluateBtn.addEventListener('click', () => {
        resolution = parseInt(gridResolutionInput.value);
        renderCPPNImages(cppns,resolution);
      });
}
/**
 * Generate a grid of normalized inputs for the CPPN.
 * @param {number} inputX - Range scale for X-axis.
 * @param {number} inputY - Range scale for Y-axis.
 * @param {number} resolution - Resolution of the grid (N x N).
 * @returns {Object} - An object containing tensors for X, Y, and R.
 */
export function generateGrid(inputX, inputY, resolution) {
    // Generate a range from -0.5 to 0.5 scaled by inputX and inputY
    const xRange = tf.linspace(-inputX / 2, inputX / 2, resolution);
    const yRange = tf.linspace(-inputY / 2, inputY / 2, resolution);
  
    // Create a 2D grid using tf.meshgrid
    const [X, Y] = tf.meshgrid(xRange, yRange);
  
    // Compute the radial distance R (optional, useful for symmetry)
    const R = tf.sqrt(tf.add(tf.square(X), tf.square(Y)));
  
    return { X, Y, R };
  }


/**
 * Render an image from the CPPN using the generated grid.
 * @param {Object} cppn - The initialized CPPN.
 * @param {number} resolution - Grid resolution (N x N).
 * @returns {ImageData} - The rendered image data.
 */
export function renderCPPNImages(cppns, resolution) {
    const canvas = document.getElementById('imageCanvas');

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Read input range values
    const inputX = parseFloat(document.getElementById('inputX').value);
    const inputY = parseFloat(document.getElementById('inputY').value);
  
    // Generate grid tensors
    const { X, Y, R } = generateGrid(inputX, inputY, resolution);
  
    // Create input tensors for the CPPN (stacked inputs)
    const inputTensors = [X.flatten(), Y.flatten(), R.flatten()];
  
    // Evaluate the CPPN to get outputs
    // for loop through cppns
    let image_x = 5;
    let image_y = 5;
    let image_width = resolution;
    let imageDatas = [];
    for (let i = 0; i < cppns.length; i++) {
        const cppn = cppns[i];
        const outputs = evaluate(cppn, inputTensors);
    
    
        // Collect RGB channels from the outputs
        const rChannel = outputs[0].reshape([resolution, resolution]);
        const gChannel = outputs[1].reshape([resolution, resolution]);
        const bChannel = outputs[2].reshape([resolution, resolution]);
    
        // Stack RGB channels and normalize for rendering
        const imageDataArray = tf.stack([rChannel, gChannel, bChannel], -1)
            .mul(255) // Scale to 0-255
            .clipByValue(0, 255) // Clamp values
            .cast('int32')
            .arraySync(); // Convert to JavaScript array for rendering
    
        // Render to the canvas
        const imageData = ctx.createImageData(resolution, resolution);
        let index = 0;
        for (let y = 0; y < resolution; y++) {
        for (let x = 0; x < resolution; x++) {
            let [r, g, b] = imageDataArray[y][x];
            if (r === undefined || g === undefined || b === undefined) {
                r = 0;
                g = 0;
                b = 0;
            }
            imageData.data[index++] = r; // Red
            imageData.data[index++] = g; // Green
            imageData.data[index++] = b; // Blue
            imageData.data[index++] = 255; // Alpha (fully opaque)
        }
        }

        ctx.putImageData(imageData, image_x, image_y);
        image_x += image_width+5;
        if (image_x + image_width > canvas.width) {
            image_x = 5;
            image_y += image_width+5;
        }
        imageDatas.push(imageData.data);

    }

    // just change the image source
    // imageElement.src = imageData;
    return imageDatas;
  }



export function initializeCanvasControls() {
  const canvas = document.getElementById('cppnCanvas');
  const ctx = canvas.getContext('2d');

  let scale = 0.90;
  let offsetX = 40;
  let offsetY = 0;
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;

  // Initialize CPPN
  const cppn = initializeCPPN(3, [3, 2], 3);

  function drawCPPN(cppn) {
    const { layers, connections } = cppn;
  
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
  
    const layerSpacing = canvas.width / (layers.length - 1);
    const nodeRadius = 10;
  
    // Helper to draw nodes
    function drawNode(x, y, label) {
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
      ctx.fillStyle = 'white';
      ctx.fill();
      ctx.strokeStyle = 'black';
      ctx.stroke();
      ctx.fillStyle = 'black';
      ctx.fillText(label, x - nodeRadius / 2, y + nodeRadius * 2);
    }
  
    // Helper to draw connections
    function drawConnection(x1, y1, x2, y2, weight) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = weight > 0 ? 'green' : 'red';
      ctx.lineWidth = Math.abs(weight) * 2;
      ctx.stroke();
    }
  
    // Calculate node positions
    const layerHeights = layers.map(
      (layer) => canvas.height / (layer.length + 1)
    );
  
    layers.forEach((layer, layerIndex) => {
      const x = layerSpacing * layerIndex;
  
      layer.forEach((node, nodeIndex) => {
        const y = layerHeights[layerIndex] * (nodeIndex + 1);
        let af = node.activationFunction;
        let afName = 'None';
        if (af !== null && af !== undefined && af.name !== undefined) {
            afName = af.name;
        }
        drawNode(x, y, `L${layerIndex}N${nodeIndex}:${afName}`);
        node.x = x; // Save position for drawing connections
        node.y = y;
      });
    });
  
    connections.forEach(({ fromNode, toNode, weight }) => {
      if (fromNode.x !== undefined && fromNode.y !== undefined && toNode.x !== undefined && toNode.y !== undefined) {
        drawConnection(fromNode.x, fromNode.y, toNode.x, toNode.y, weight.arraySync());
      }
    });
  
    ctx.restore();
  }

  // Event Listeners
  canvas.addEventListener('mousedown', (event) => {
    isDragging = true;
    lastMouseX = event.clientX;
    lastMouseY = event.clientY;
  });

  canvas.addEventListener('mousemove', (event) => {
    if (isDragging) {
      const dx = event.clientX - lastMouseX;
      const dy = event.clientY - lastMouseY;
      offsetX += dx;
      offsetY += dy;
      lastMouseX = event.clientX;
      lastMouseY = event.clientY;

      drawCPPN(cppn);
    }
  });

  canvas.addEventListener('mouseup', () => {
    isDragging = false;
  });

  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const zoomAmount = -event.deltaY * 0.001;
    const newScale = Math.min(Math.max(0.5, scale + zoomAmount), 5);

    const mouseX = event.clientX - canvas.offsetLeft;
    const mouseY = event.clientY - canvas.offsetTop;

    offsetX -= (mouseX / scale) * (newScale - scale);
    offsetY -= (mouseY / scale) * (newScale - scale);

    scale = newScale;
    drawCPPN(cppn);
  });

  // Initial draw
  drawCPPN(cppn);
}
