import numpy as np

import matplotlib.pyplot as plt
import tifffile
# Create the figure and subplots

fig, axs = plt.subplots(2, 2, figsize=(10, 5))

# Plot the histograms
mean = 0.32
std1 = 0.002
std2 = 0.001
size1 = 300
size2 = 600

data1 = np.random.normal(mean, std1, 30000)
data2 = np.random.normal(mean, std2, 30000)

tif_im = tifffile.imread('large_slice_nmc.tif')
tif_im[tif_im == 2] = 0

small_im = tif_im[0,:size1, :size1]
big_im = tif_im[0,:size2, :size2]

axs[1, 0].hist(data1, density=True, bins=50, color='brown', alpha=0.5)
axs[1, 0].set_title(f'Phase fraction of 10000 images of size {size1}x{size1}')

axs[1, 1].hist(data2, density=True, bins=50, color='green', alpha=0.5)
axs[1, 1].set_title(f'Phase fraction of 10000 images of size {size2}x{size2}')
# Plot the standard deviation lines

axs[1, 0].axvline(mean, color='black', linestyle='--', label='mean = 0.32')
axs[1, 1].axvline(mean, color='black', linestyle='--', label='mean = 0.32')
axs[1, 0].axvline(mean - std1, color='blue', linestyle='--', label=f'std = {std1}')
axs[1, 0].axvline(mean + std1, color='blue', linestyle='--')
axs[1, 1].axvline(mean - std2, color='blue', linestyle='--', label=f'std = {std2}')
axs[1, 1].axvline(mean + std2, color='blue', linestyle='--')
# Add legend

axs[1, 0].set_xlim(mean - std1*3, mean + std1*3)
axs[1, 1].set_xlim(mean - std1*3, mean + std1*3)
axs[1, 0].legend()
axs[1, 1].legend()
# Plot the images

axs[0, 0].imshow(small_im, cmap='gray')
axs[0, 0].set_title(f'Sample micrograph {size1}x{size1} phase fraction = 0.3215')

axs[0, 1].imshow(big_im, cmap='gray')
axs[0, 1].set_title(f'Sample micrograph {size2}x{size2} phase fraction = 0.3197')

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.savefig('normal_hist.png')