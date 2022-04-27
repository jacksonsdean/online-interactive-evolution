"""To test the backend interactively"""
#%%
import json
from multiprocessing import Process
import time
import matplotlib.pyplot as plt
from nextGeneration.lambda_function import lambda_handler
from local_test import run_server
from test import TEST_RESET_EVENT
from skimage.color import hsv2rgb
import numpy as np
from nextGeneration.activation_functions import gauss, identity, sin, tanh
from nextGeneration.config import Config
from nextGeneration.cppn import CPPN, Node, NodeType

def show_images(imgs, color_mode="L", titles=[], height=10):
    """Show an array of images in a grid"""
    num_imgs = len(imgs)
    fig = plt.figure(figsize=(20, height))
    for i, image in enumerate(imgs):
        ax = fig.add_subplot(num_imgs//5 +1, 5, i+1)
        if(len(titles)> 0):
            ax.set_title(titles[i])
        else:
            ax.set_title(f"{i}")
        show_image(image, color_mode)
    plt.show()

def show_image(img, color_mode, ax = None):
    """Show an image"""
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if(color_mode == 'L'):
        if(ax==None):
            plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)
        else:
            ax.imshow(img, cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)

    elif(color_mode == "HSL"):
        img = hsv2rgb(img)
        if(ax==None):
            plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)
        else:
            ax.imshow(img)
    else:
        if(ax==None):
            plt.imshow(img,cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)
        else:
            ax.imshow(img)

#%%
config = Config()
cppn = CPPN(config)
image_data = cppn.get_image_data_fast_method(32,32)
print(np.min(image_data), np.max(image_data))

plt.imshow(image_data, cmap='gray', vmin = -1, vmax = 1)
plt.show()
config.color_mode = "RGB"
config.num_outputs = 3
ims = []
for i in range(10):
    cppn_color = CPPN(config)
    image_data = cppn_color.get_image_data_fast_method(32,32)
    ims.append(image_data)
show_images(ims, color_mode="RGB")
plt.show()
#%%
node_genome = []
id = 0
layer = 0
node_genome.append(Node(identity, NodeType.INPUT, id, layer)); id+=1
node_genome.append(Node(identity, NodeType.INPUT, id, layer)); id+=1
node_genome.append(Node(identity, NodeType.INPUT, id, layer)); id+=1
node_genome.append(Node(identity, NodeType.HIDDEN, id, layer)); id+=1
layer=2
node_genome.append(Node(tanh, NodeType.OUTPUT, id, layer)); id+=1
node_genome.append(Node(tanh, NodeType.OUTPUT, id, layer)); id+=1
node_genome.append(Node(tanh, NodeType.OUTPUT, id, layer)); id+=1
layer=1
node_genome.append(Node(gauss, NodeType.HIDDEN, id, layer)); id+=1
node_genome.append(Node(sin, NodeType.HIDDEN, id, layer)); id+=1

config = Config()
config.color_mode = "RGB"
cppn = CPPN(config, nodes=node_genome)
print(cppn.connection_genome)
img = cppn.get_image_data_fast_method(32,32)
show_image(img, color_mode="RGB")
plt.show()
#%%
import requests

# server_process = Process(target=run_server, args=())
# server_process.start()
timeout = 10  # wait for 10 seconds before failing
server_response = False
event = TEST_RESET_EVENT
while timeout > 0 and not server_response:
    response = requests.post("http://localhost:5000", json=event)
    server_response = response.status_code == 200
    time.sleep(1)
    timeout -= 1
imgs = []
obj = response.json()["body"]
config = Config.create_from_json(obj["config"])
for indiv in obj["population"]:
    cppn = CPPN.create_from_json(indiv, config)
    img = cppn.get_image_data_fast_method(32,32)
    imgs.append(img)
show_images(imgs, color_mode="RGB")
# server_process.terminate()  # kill test server

