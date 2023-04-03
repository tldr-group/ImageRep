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

def func(x, a, b, c):
    return a*np.e**(-b*x) + c
    # return a/(x+c) + b

def func2(x, a, b):
    return a*x + b

with open("data.json", "r") as fp:
    data = json.load(fp)['generated_data'] 

l=len(list(data.keys()))
c=[(0,0,0), (0.5,0.5,0.5)]
fig, axs = plt.subplots(l, 5)
fig.set_size_inches(25, l*4)
preds = [[],[]]
facs = [[],[]]
sas =[]
for i, n in enumerate(list(data.keys())[:l]):
    img = tifffile.imread(f'D:/Dataset/{n}/{n}.tif')
    d = data[n]
    d1 = data[n]

    axs[i,0].set_title(n)
    for j, met in enumerate(['vf', 'sa']):
        axs[i, j*2].plot(d['ls'], d[f'err_exp_{met}'], c='black')
        axs[i, j*2].plot(d[f'ls_model_{met}'], d[f'err_model_{met}'], c='black') 
        y = d[f'tpc_{met}']
        cut = max(20,np.argmin(y))
        y = y[:cut]
        x = np.linspace(1, len(y), len(y))
        bounds = ((-np.inf, 0.01, -np.inf), (np.inf, np.inf, np.inf))
        try:
            coefs_poly3d, _ = curve_fit(func, x, y, bounds=bounds)
            y_data = func(x,*coefs_poly3d)
        except:
            print(met, n)
            preds[1-j].pop(-1)
            facs[1-j].pop(-1)
            # y_data = y
            continue
        axs[i, j*2+1].plot(x, y_data, c='blue')
        axs[i, j*2+1].plot(x, y, c='blue', ls='--')
        kneedle = kneed.KneeLocator(x, y_data, S=1.0, curve="convex", direction="decreasing")
        axs[i, j*2+1].axvline(kneedle.knee, c = 'b')
        axs[i, j*2+1].axvline(d[f'fac_{met}'], c = 'g')
        preds[j].append(kneedle.knee)
        facs[j].append(d1[f'fac_{met}'])
    sas.append(d['sa'])
    axs[i, 4].imshow((img[0]*0.5)+0.5, cmap='gray')

data
# with open("data5", "w") as fp:
#     json.dump(data,fp) 

plt.tight_layout()

fig, axs = plt.subplots(1,2)
fig.set_size_inches(10,5)
for i in range(2):
    ax = axs[i]
    y=np.array(preds[i])
    targ = np.array(facs[i])
    ax.plot(np.arange(60), np.arange(60), c='g')
    ax.scatter(targ, y, s=5, c='b')
    coefs_poly3d, _ = curve_fit(func2, y, targ)
    y_data = func2(np.arange(60),*coefs_poly3d)
    ax.plot(y_data, np.arange(60), c='r')
    y_data = func2(y,*coefs_poly3d)

    ax.set_aspect(1)
    ax.set_xlabel('Fac from model')
    label = 'True fac' if i==0 else 'SA fac'
    ax.set_ylabel(label)
    ax.set_xlim(0,60)
    ax.set_ylim(0,60)
    res = np.mean(abs(y-targ)/targ)*100
    err = abs(y_data-targ)/targ
    res2 = np.mean(err)*100
    idx = np.argpartition(err, -3)[-3:]
    for j in idx:
        print(list(data.keys())[j])
    # res = 100*(np.mean((y-targ)**2/targ))**0.5

    print(res,res2)
# plt.savefig('small.png')





