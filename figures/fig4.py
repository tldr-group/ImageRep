import tifffile
import json
import numpy as np
from scipy.optimize import curve_fit
import util
import os
import matplotlib.pyplot as plt

with open("data2.json", "r") as fp:
    data = json.load(fp)['validation_data']
projects = os.listdir('D:\Dataset')
# projects = [f'D:/Dataset/{p}/{p}' for p in projects]

fig, axs = plt.subplots(1,2)
fig.set_size_inches(10,5)
err_exp_vf = np.array([data[n]['err_exp_vf'] for n in data.keys()])
err_exp_sa = np.array([data[n]['err_exp_sa'] for n in data.keys()])
pred_err_vf = np.array([data[n]['pred_err_vf'] for n in data.keys()])
pred_err_sa = np.array([data[n]['pred_err_sa'] for n in data.keys()])

axs[0].scatter(err_exp_vf, pred_err_vf)
axs[1].scatter(err_exp_sa[pred_err_sa!=0], pred_err_sa[pred_err_sa!=0])

for ax, met in zip(axs.ravel(), ['Volume fraction', 'Surface area']):
    ax.plot(np.arange(20), np.arange(20))
    ax.set_xlabel('Error from statistical analysis')
    ax.set_ylabel('Error from tpc analysis')
    ax.set_title(met)
    ax.set_aspect(1)
    ax.set_yticks([10, 5, 10, 15])




