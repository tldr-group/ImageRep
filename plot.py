import os
import matplotlib.pyplot as plt
import tifffile
import json
import numpy as np
import kneed
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a*np.e**(-b*x) + c

with open("data_tpc", "r") as fp:
    data_tpc = json.load(fp) 

l=76
c=[(0,0,0), (0.5,0.5,0.5)]
fig, axs = plt.subplots(l, 4)
fig.set_size_inches(16, l*4)
pred = []
facs = []
for i, n in enumerate(list(data_tpc.keys())[:l]):
    img = tifffile.imread(f'D:/Dataset/{n}/{n}.tif')
    axs[i, 3].imshow((img[0]*0.5)+0.5, cmap='gray')
    for j in range(2):
        # plot the fitted err curves as func of l
        d = data_tpc[n][str(j)]
        axs[i, j].plot(d['ls'], d['err_exp'], c=c[j])
        axs[i, j].plot(d['ls_model'], d['err_model'], c=c[j])
    # now we try do a prediction of fac using tpc
    # first plot tpc
    axs[i, 2].plot(d['tpc'][:150], c='black')
    # fit a exp decay to the tpc
    y = np.array(d['tpc'])
    x = np.linspace(1, 149, 149)
    coefs_poly3d, _ = curve_fit(func, x, y)
    y_data = func(x, *coefs_poly3d)
    axs[i, 2].plot(x, y_data, c='black', ls='--')
    # find the knee
    kneedle = kneed.KneeLocator(x, y_data, S=1.0, curve="convex", direction="decreasing")
    # plot knee of tpc, and also the fitted fac from model
    axs[i, 2].axvline(kneedle.knee, c = 'green')
    axs[i, 2].axvline(d['fac'], c = 'red')
    facs.append(d['fac'])
    pred.append(round(kneedle.knee, 3))

plt.tight_layout()

fig, ax = plt.subplots(1,1)
ax.plot(np.arange(60), np.arange(60))
ax.scatter(facs, pred, s=1, c='black')
ax.set_aspect(1)
ax.set_xlabel('fac from model')
ax.set_ylabel('fac from tpc only')



