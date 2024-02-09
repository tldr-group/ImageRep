import tifffile
import json
import numpy as np
from scipy.optimize import curve_fit
import util
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import math

# with open("data3d.json", "r") as fp:
#     data3d = json.load(fp)['validation_data3d']

# with open("data2d.json", "r") as fp:
#     data2d = json.load(fp)['validation_data']

with open("datafin_sg1_fft.json", "r") as fp:
    datafull = json.load(fp)

def objective_function(beta, y_true, y_pred):
    y_pred = beta[0]*y_pred + beta[1]
    return(np.mean(np.abs((y_true - y_pred) / (y_true+y_pred))) * 100)

# a = 2.5
# Slope and intercept for tpc to stat.analysis error fit, that have 0 mean: 
slope_and_intercept = {'2D':
                       {'Volume fraction': (0.925, 0), 'Surface area': (0.925, 0)},
                       '3D':
                       {'Volume fraction': (1, 0), 'Surface area': (1, 0)}
}

periodic_micros = np.load('periodicity_list.npy')
# You must provide a starting point at which to initialize
# the parameter search space

with open("micro_names.json", "r") as fp:
    micro_names = json.load(fp)
projects = [f'/home/amir/microlibDataset/{p}/{p}' for p in micro_names]
# projects = [f'D:/Dataset/{p}/{p}' for p in projects]
fig, axs = plt.subplots(2,2)
fig.set_size_inches(10,10)
dims = ['2D', '3D']
for i, data in enumerate(dims):
    # data.pop('microstructure054')
    data_dim = datafull[f'validation_data{data}']
    err_exp_vf = np.array([data_dim[n]['exp_ir_vf'] for n in data_dim.keys()])
    err_exp_sa = np.array([data_dim[n]['exp_err_vf'] for n in data_dim.keys()])
    pred_err_vf = np.array([data_dim[n]['pred_ir_vf'] for n in data_dim.keys()])
    pred_err_sa = np.array([data_dim[n]['pred_err_vf'] for n in data_dim.keys()])
    data_2d = datafull[f'validation_data2D']
    dim_var = np.array([data_2d[n]['dim_variation'] if 'dim_variation' in data_2d[n] else 0 for n in data_2d.keys()])
    # dim_var = np.log(dim_var)
    dim_var = dim_var/np.max(dim_var) if np.max(dim_var) > 0 else dim_var
    # colors = np.array([(0,0,0) if n in periodic_micros else (1,0,0) for n in data.keys()])
    colors = np.array([(dim_var[i],0,0) for i in range(len(dim_var))])
    vf_results = [err_exp_vf, pred_err_vf]
    sa_results = [[err_exp_sa[0]], [pred_err_sa[0]]]
    for idx in range(1, len(err_exp_sa)):
        if not math.isnan(err_exp_sa[idx]) and not math.isnan(pred_err_sa[i]):
            sa_results[0].append(err_exp_sa[idx])
            sa_results[1].append(pred_err_sa[idx])
    sa_results[0], sa_results[1] = np.array(sa_results[0]), np.array(sa_results[1])
    # sa_results = [err_exp_sa[pred_err_sa!=math.isnan], pred_err_sa[pred_err_sa!=math.nan]]
    for j, (met, res) in enumerate(zip(['Volume fraction', 'Surface area'], [vf_results, sa_results])):

        
        ax = axs[i, j]
        end_idx = 78
        res[0], res[1] = res[0][:end_idx], res[1][:end_idx]
        colors = colors[:end_idx]
    
        beta_init = np.array([1, 0])
        # bounds = [(-10,10),(-10,10)]
        # To find a good fit:
        # result = minimize(objective_function, beta_init, args=(res[0], res[1]),
                #   options={'maxiter': 50000})
        # print(result)
        # y_data = result.x[0]*res[1] + result.x[1]

        # if data == '3D':
        #     res[1] = res[1]**(1/3)
        # else:
        #     res[1] = np.sqrt(res[1])

        slope, intercept = slope_and_intercept[dims[i]][met]
        y_data = slope*res[1] + intercept
        
        

        without_last_outlier = np.logical_and(y_data > 0, res[0] > 0)
        ax.scatter(res[0][without_last_outlier], y_data[without_last_outlier], s=7, label='Predictions', c=colors[without_last_outlier])
        
        # print(f'slope = {slope} and intercept = {intercept}')
        ax.set_xlabel(f'IR from stat. analysis')
        ax.set_ylabel(f'IR from TPC')
        ax.set_title(f'{met} {dims[i]}')
        
        errs = (res[0]-y_data)/y_data 
        
        max_val = int(np.max([np.max(y_data[without_last_outlier]),np.max(res[0][without_last_outlier])]))+2
        
        x = np.arange(max_val+1)
        ax.plot(x, x, label = 'Ideal error prediction', color='black')
        ax.set_yticks(np.arange(0, max_val, 5))
        ax.set_xticks(np.arange(0, max_val, 5))
        ax.set_xlim(right=max_val)
        ax.set_ylim(top=max_val)
        errs = np.sort(errs) 
        std = np.std((y_data-res[0])/y_data) 
        
        z = norm.interval(0.9)[1]
        err = std*z
        print(f'{met} {dims[i]} std = {std}')
        print(f'mean = {np.mean(errs)}')
        print(f'mape = {np.mean(np.abs(errs))}')
        print(f'error = {err}')
        
        ax.plot(np.arange(max_val), np.arange(max_val), c='black')
        ax.plot(x ,x/(1+err), c='black', ls='--', linewidth=1)
        fill_1 = ax.fill_between(x, np.ones(x.shape[0])*(max_val),x/(1+err), alpha=0.2, label = f'95% confidence range')
        ax.set_aspect('equal')
        
        # print(f'Unfitted {met} MAPE error {dims[i]}: {abs((res[0] - res[1])/res[0]).mean()}')
        # print(f'Fitted {met} MAPE error {dims[i]}: {abs((res[0] - y_data)/res[0]).mean()}')
axs[0, 0].legend()  # TODO add text to the legend indicating the MAPE
fig.savefig('fig3.pdf', format='pdf')         



