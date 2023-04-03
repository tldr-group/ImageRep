import os
import matplotlib.pyplot as plt
import tifffile
import json
import numpy as np
import kneed
from scipy.optimize import curve_fit
from scipy import stats
import torch
from .. import util
from torch.nn.functional import interpolate

def sample_error(img, ls, vf=0.5):
    err = []
    for l in ls:
        vfs = torch.mean(img[:,:l,:l], dim=(1,2))
        std = torch.std(vfs.cpu())
        err.append(100*((1.96*std)/0.5))
    return err

errs = []
berns = []

ls = torch.arange(300,800, 20)

img1 = torch.randint(0,2,(2000, 1, ls[-1],ls[-1]), device = torch.device('cuda:0')).float()
errs.append(sample_error(img1[:,0], ls))
berns.append(util.bernouli(0.5, ls))

img2 = interpolate(img1[:,:,:ls[-1]//2, :ls[-1]//2], scale_factor=(2,2), mode='nearest')
errs.append(sample_error(img2[:,0], ls))
ls2 = [l/2 for l in ls]
berns.append(util.bernouli(0.5, ls2))

with open("data.json", "r") as fp:
    data = json.load(fp)['generated_data'] 

n = 'microstructure630'
img3 = tifffile.imread(f'D:/Dataset/{n}/{n}.tif')
d = data[n]
errs.append(d[f'err_exp_vf'][::4])
ls3 = [l/d['fac_vf'] for l in ls]
berns.append(util.bernouli(d['vf'], ls3))

fig, axs = plt.subplots(2,2)
fig.set_size_inches(6,6)
l=150
axs[0,0].imshow(img1[0,0, :l, :l].cpu(), cmap='gray')
axs[0,1].imshow(img2[0,0, :l, :l].cpu(), cmap='gray')
axs[1,0].imshow(img3[0, :150, :150], cmap='gray')

cs = ['r', 'b', 'g']
for err, bern, c in zip(errs, berns, cs):
    axs[1,1].scatter(ls, err, c=c, s=8,marker = 'x')
    axs[1,1].plot(ls, bern, lw=1, c=c)

plt.tight_layout()
for ax in axs.ravel()[:3]:
    ax.set_xticks([])
    ax.set_yticks([])
                  


