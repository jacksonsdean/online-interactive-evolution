"""To test the backend interactively"""
#%%
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb
import numpy as np
from activation_functions import gauss, identity, sawtooth, tanh
from config import Config
from cppn import CPPN, Node, NodeType


def show_images(imgs, color_mode="L", titles=[], height=10):
    num_imgs = len(imgs)
    # plt.figure(figsize=(20, num_imgs //1.8))
    # plt.figure(figsize=(20, num_imgs*1))
    # columns = num_imgs//4
    # rows = 4 
    fig = plt.figure(figsize=(20, height))
    # fig.subplots_adjust(hspace=0.4, wspace=0.3)
    for i, image in enumerate(imgs):
        ax = fig.add_subplot(num_imgs//5 +1, 5, i+1)
        if(len(titles)> 0):
            ax.set_title(titles[i])
        else:
            ax.set_title(f"{i}")
        show_image(image, color_mode)
    plt.show()

def show_image(img, color_mode, ax = None):
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if(color_mode == 'L'):
        if(ax==None):
            plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

    elif(color_mode == "HSL"):
        img = hsv2rgb(img)
        if(ax==None):
            plt.imshow(img, vmin=-1, vmax=1)
        else:
            ax.imshow(img)
    else:
        if(ax==None):
            plt.imshow(img, vmin=-1, vmax=1)
        else:
            ax.imshow(img)
            
    # if(ax==None):
    #     plt.show()
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
node_genome.append(Node(sawtooth, NodeType.HIDDEN, id, layer)); id+=1

config = Config()
config.color_mode = "RGB"
cppn = CPPN(config, nodes=node_genome)
print(cppn.connection_genome)
img = cppn.get_image_data_fast_method(32,32)
print(img)
show_image(img, color_mode="RGB")
plt.show()
