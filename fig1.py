import os
import matplotlib.pyplot as plt
import tifffile
import json
import numpy as np
import kneed
from scipy.optimize import curve_fit
from scipy import stats
import torch
import util
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
img_dims = [np.array((l, l)) for l in ls]
ns = util.ns_from_dims(img_dims, 1)
img1 = torch.randint(0,2,(2000, 1, ls[-1],ls[-1]), device = torch.device('cuda:0')).float()
errs.append(sample_error(img1[:,0], ls))
berns.append(util.bernouli(0.5, ns))

img2 = interpolate(img1[:,:,:ls[-1]//2, :ls[-1]//2], scale_factor=(2,2), mode='nearest')
errs.append(sample_error(img2[:,0], ls))
ns = util.ns_from_dims(img_dims, 2)
berns.append(util.bernouli(0.5, ns))

with open("datafin.json", "r") as fp:
    data = json.load(fp)['generated_data'] 

n = 'microstructure205'
img3 = tifffile.imread(f'D:/Dataset/{n}/{n}.tif')
d = data[n]
errs.append(d[f'err_exp_vf'][::4])
ns = util.ns_from_dims(img_dims, d['fac_vf'])
berns.append(util.bernouli(d['vf'], ns))

fig, axs = plt.subplots(2,2)
fig.set_size_inches(8,8)
l=150
axs[0,0].imshow(img1[0,0, :l, :l].cpu(), cmap='gray')
axs[0,1].imshow(img2[0,0, :l, :l].cpu(), cmap='gray')
axs[1,0].imshow(img3[0, :150, :150], cmap='gray')

cs = ['r', 'b', 'g']
labs = ['a) random 1x1', 'b) random 2x2', 'c) microstructure 205']
for l, err, bern, c in zip(labs, errs, berns, cs):
    axs[1,1].scatter(ls, err, c=c, s=8,marker = 'x')
    axs[1,1].plot(ls, bern, lw=1, c=c, label=l)
axs[1,1].legend()
axs[1,1].set_xlabel('Error from statistical analysis (%)')
axs[1,1].set_ylabel('Error from tpc analysis (%)')
plt.tight_layout()
for ax in axs.ravel()[:3]:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig('fig1.pdf', format='pdf')         


