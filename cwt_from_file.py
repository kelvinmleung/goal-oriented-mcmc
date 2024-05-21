import numpy as np
import matplotlib.pyplot as plt
import seaborn
from PIL import Image

im = Image.open('SHIFT_AVNG_Canopy_WaterContent_2242/data/20220224_cwc_phase.tif')
imarray = np.array(im)[5000:5050, 5000:5050]

plt.figure()
seaborn.heatmap(imarray)
plt.show()