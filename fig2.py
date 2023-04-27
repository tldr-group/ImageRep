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

with open("datafin.json", "r") as fp:
    data = json.load(fp)['generated_data'] 

l=len(list(data.keys()))
# l=3
c=[(0,0,0), (0.5,0.5,0.5)]
# plotting = [f'microstructure{f}' for f in [235, 209,205,177]]
plotting = [f'microstructure{f}' for f in [722, 235,205,177]]

# plotting = [k for k in data.keys()]
l = len(plotting)
fig, axs = plt.subplots(l, 3)
fig.set_size_inches(12, l*4)
preds = [[],[]]
facs = [[],[]]
sas =[]
i=0
for n in list(data.keys()):
    if n not in plotting:
        continue
    img = tifffile.imread(f'D:/Dataset/{n}/{n}.tif')
    d = data[n]
    d1 = data[n]
    
    csets = [['black', 'black'], ['gray', 'gray']]
    for j, met in enumerate(['vf', 'sa']):
        cs = csets[j]
        axs[i, 1].plot(d['ls'], d[f'err_exp_{met}'], c=cs[0], label = f'{met} errors from sampling')
        axs[i, 1].plot(d[f'ls'], d[f'err_model_{met}'], c=cs[0], ls='--', label = f'{met} errors from bernouli') 
        y = d[f'tpc_{met}']
        cut = max(20,np.argmin(y))
        x = np.linspace(1, len(y), len(y))
        bounds = ((-np.inf, 0.01, -np.inf), (np.inf, np.inf, np.inf))
        try:
            coefs_poly3d, _ = curve_fit(util.tpc_fit, x[:cut], y[:cut], bounds=bounds)
            y_data = util.tpc_fit(x[:cut],*coefs_poly3d)
        except:
            print(met, n)
            preds[1-j].pop(-1)
            facs[1-j].pop(-1)
            # y_data = y
            continue
        y = np.array(y)
        y_data = np.array(y_data)
        axs[i, 2].plot(x[:cut], y_data/y.max(), c=cs[1], label=f'{met} fitted tpc')
        axs[i, 2].plot(x, y/y.max(), c=cs[1], ls='--', label=f'{met} raw tpc')
        knee = int(kneed.KneeLocator(x[:cut], y_data, S=1.0, curve="convex", direction="decreasing").knee)
        axs[i, 2].scatter(x[knee], y_data[knee]/y.max(), c =cs[1], marker = 'x', label=f'{met} fac from tpc', s=100)
        fac = d[f'fac_{met}']
        axs[i, 2].scatter(x[int(fac)], y_data[int(fac)]/y.max(), facecolors='none', edgecolors = cs[1], label=f'{met} fac from sampling', s=100)
        preds[j].append(knee)
        facs[j].append(fac)
        if i ==0:
            axs[i,1].legend()
            axs[i,2].legend()
            axs[i,1].set_xlabel('Error from statistical analysis (%)')
            axs[i,1].set_ylabel('Error from tpc analysis (%)')
    fac = int(d[f'fac_vf'])
    sas.append(d['sa'])
    im = img[0]*255
    si_size, nfacs = 160, 5
    sicrop = int(fac*nfacs)
    print(fac, sicrop)
    subim=torch.tensor(im[-sicrop:,-sicrop:]).unsqueeze(0).unsqueeze(0).float()
    subim = interpolate(subim, size=(si_size,si_size), mode='nearest')[0,0]
    subim = np.stack([subim]*3, axis=-1)
    
    subim[:5,:,:] = 125
    subim[:,:5,:] = 125
    # subim[5:20, 5:50, :]
    subim[10:15, 10:10+si_size//nfacs, :] = 0
    subim[10:15, 10:10+si_size//nfacs, 1:-1] = 125

    im = np.stack([im]*3, axis=-1)
    im[-si_size:,-si_size:] = subim
    axs[i, 0].imshow(im)
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([])
    axs[i, 0].set_ylabel(f'M{n[1:]}')
    axs[i, 0].set_xlabel(f'Fac: {fac}   Inset mag: x{np.round(si_size/sicrop, 2)}')

    i+=1


plt.tight_layout()
plt.savefig('fig2.pdf', format='pdf')         

# fig, axs = plt.subplots(1,2)
# fig.set_size_inches(10,5)
# for i in range(2):
#     ax = axs[i]
#     y=np.array(preds[i])
#     targ = np.array(facs[i])
#     ax.plot(np.arange(60), np.arange(60), c='g')
#     ax.scatter(targ, y, s=5, c='b')
#     coefs_poly3d, _ = curve_fit(util.linear_fit, y, targ)
#     y_data = util.linear_fit(np.arange(60),*coefs_poly3d)
#     ax.plot(y_data, np.arange(60), c='r')
#     y_data = util.linear_fit(y,*coefs_poly3d)

#     ax.set_aspect(1)
#     ax.set_xlabel('Fac from model')
#     label = 'True fac' if i==0 else 'SA fac'
#     ax.set_ylabel(label)
#     ax.set_xlim(0,60)
#     ax.set_ylim(0,60)
#     res = np.mean(abs(y-targ)/targ)*100
#     err = abs(y_data-targ)/targ
#     res2 = np.mean(err)*100
#     idx = np.argpartition(err, -3)[-3:]
#     for j in idx:
#         print(list(data.keys())[j])
#     # res = 100*(np.mean((y-targ)**2/targ))**0.5

#     print(res,res2)





