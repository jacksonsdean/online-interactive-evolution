"""To test the backend interactively"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from cppn import CPPN
#%%
config = Config()
cppn = CPPN(config)
image_data = cppn.get_image_data_fast_method(32,32,"L")
print(np.min(image_data), np.max(image_data))
plt.imshow(image_data, cmap='gray', vmin = -1, vmax = 1)
plt.show()