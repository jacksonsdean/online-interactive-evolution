"""To test the backend interactively"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from cppn import CPPN
#%%
config = Config()
cppn = CPPN(config)
image_data = cppn.get_image_data_fast_method(32,32)
print(np.min(image_data), np.max(image_data))

plt.imshow(image_data, cmap='gray', vmin = -1, vmax = 1)
plt.show()
config.color_mode = "RGB"
cppn_color = CPPN(config)
image_data = cppn_color.get_image_data_fast_method(32,32)
print(image_data.shape)
plt.imshow(image_data)
plt.show()