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

with open("data.json", "r") as fp:
    data = json.load(fp)['generated_data'] 

l=len(list(data.keys()))
# l=3
c=[(0,0,0), (0.5,0.5,0.5)]
fig, axs = plt.subplots(l, 3)
fig.set_size_inches(12, l*4)
preds = [[],[]]
facs = [[],[]]
sas =[]
for i, n in enumerate(list(data.keys())[:l]):
    img = tifffile.imread(f'D:/Dataset/{n}/{n}.tif')
    d = data[n]
    d1 = data[n]

    axs[i,0].set_title(n)
    csets = [['black', 'black'], ['gray', 'gray']]
    for j, met in enumerate(['vf', 'sa']):
        cs = csets[j]
        axs[i, 0].plot(d['ls'], d[f'err_exp_{met}'], c=cs[0], label = f'{met} errors from sampling')
        axs[i, 0].plot(d[f'ls_model_{met}'], d[f'err_model_{met}'], c=cs[0], ls='--', label = f'{met} errors from bernouli') 
        y = d[f'tpc_{met}']
        cut = max(20,np.argmin(y))
        y = y[:cut]
        x = np.linspace(1, len(y), len(y))
        bounds = ((-np.inf, 0.01, -np.inf), (np.inf, np.inf, np.inf))
        try:
            coefs_poly3d, _ = curve_fit(util.tpc_fit, x, y, bounds=bounds)
            y_data = util.tpc_fit(x,*coefs_poly3d)
        except:
            print(met, n)
            preds[1-j].pop(-1)
            facs[1-j].pop(-1)
            # y_data = y
            continue
        y = np.array(y)
        y_data = np.array(y_data)
        axs[i, 1].plot(x, y_data/y.max(), c=cs[1], label=f'{met} raw tpc')
        axs[i, 1].plot(x, y/y.max(), c=cs[1], ls='--', label=f'{met} fitted tpc')
        knee = int(kneed.KneeLocator(x, y_data, S=1.0, curve="convex", direction="decreasing").knee)
        axs[i, 1].scatter(x[knee], y_data[knee]/y.max(), c =cs[1], marker = 'x', label=f'{met} fac from tpc', s=100)
        fac = d[f'fac_{met}']
        axs[i, 1].scatter(x[int(fac)], y_data[int(fac)]/y.max(), facecolors='none', edgecolors = cs[1], label=f'{met} fac from sampling', s=100)
        preds[j].append(knee)
        facs[j].append(fac)
        if i ==0:
            axs[i,0].legend()
            axs[i,1].legend()

    sas.append(d['sa'])
    axs[i, 2].imshow((img[0]*0.5)+0.5, cmap='gray')


plt.tight_layout()

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
# plt.savefig('small.png')





