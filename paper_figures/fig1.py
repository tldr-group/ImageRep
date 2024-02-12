import os
import matplotlib.pyplot as plt
import tifffile
import json
import numpy as np
import torch
import util
from torch.nn.functional import interpolate

def sample_error(img, ls, vf=0.5):
    err = []
    dims = len(img.shape) - 1
    for l in ls:
        if dims == 1:
            vfs = torch.mean(img[:, :l*l], dim=(1))
        elif dims == 2:  
            vfs = torch.mean(img[:, :l, :l], dim=(1, 2))
        else:  # 3D
            new_l = int(l**(2/3))
            vfs = torch.mean(img[:, :new_l, :new_l, :new_l], dim=(1, 2, 3))
        std = torch.std(vfs.cpu())
        err.append(100*((1.96*std)/vf))  # percentage of error from 0.5
    return err

errs = []
berns = []



with open("data_gen.json", "r") as fp:
    data = json.load(fp)['generated_data'] 

n = 'microstructure205'
img3 = tifffile.imread(f'/home/amir/microlibDataset/{n}/{n}.tif')
d = data[n]


ls = torch.arange(300,800, 20)
img_dims = [np.array((l, l)) for l in ls]
ns = util.ns_from_dims(img_dims, 1)
img1 = torch.rand((1000, 1, ls[-1],ls[-1]), device = torch.device('cuda:0')).float()
img1[img1<=d['vf']] = 0
img1[img1>d['vf']] = 1
img1 = 1-img1
errs.append(sample_error(img1[:,0], ls, d['vf']))
berns.append(util.bernouli(d['vf'], ns))

img2 = interpolate(img1[:,:,:ls[-1]//2, :ls[-1]//2], scale_factor=(2,2), mode='nearest')
errs.append(sample_error(img2[:,0], ls, d['vf']))
ns = util.ns_from_dims(img_dims, 2)
berns.append(util.bernouli(d['vf'], ns))

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
labs = ['a) random 1x1', 'b) random 2x2', 'c) micrograph 205']
for l, err, bern, c in zip(labs, errs, berns, cs):
    axs[1,1].scatter(ls, err, c=c, s=8,marker = 'x')
    axs[1,1].plot(ls, bern, lw=1, c=c, label=l)
axs[1,1].legend()
axs[0,0].set_title('a) Random 1x1 (vf = 0.845, $a_2$ = 1)')
axs[0,1].set_title('b) Random 2x2 (vf = 0.845, $a_2$ = 2)')
axs[1,0].set_title('c) Micro. 205 (vf = 0.845, $a_2$ = 7.526)')
axs[1,1].set_title('d) Experimental integral range fit')
axs[1,1].set_xlabel('Image length size [pixels]')
axs[1,1].set_ylabel('Volume fraction percentage error [%]')
axs[1,1].set_ylim(bottom=0)
plt.tight_layout()
for ax in axs.ravel()[:3]:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig('fig1.pdf', format='pdf')         

