import os
import matplotlib.pyplot as plt
import tifffile
import json
with open("data", "r") as fp:
    data = json.load(fp) 
l=2
fig, axs = plt.subplots(l, 2)
fig.set_size_inches(16, l*8)
for i, n in enumerate(list(data.keys())[:l]):
    img = tifffile.imread(f'E:/Dataset/{n}/{n}.tif')
    d = data[n]
    axs[i, 0].plot(d['ls'], d['err_exp'])
    axs[i, 0].plot(d['ls_model'], d['err_model'])
    axs[i, 1].imshow(img[0])
plt.tight_layout()


