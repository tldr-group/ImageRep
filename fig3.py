import tifffile
import json
import numpy as np
from scipy.optimize import curve_fit
import util
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# with open("data3d.json", "r") as fp:
#     data3d = json.load(fp)['validation_data3d']

# with open("data2d.json", "r") as fp:
#     data2d = json.load(fp)['validation_data']

with open("datafin.json", "r") as fp:
    datafull = json.load(fp)

def objective_function(beta, y_true, y_pred):
    y_pred = beta[0]*y_pred + beta[1]
    return(np.mean(np.abs((y_true - y_pred) / (y_true+y_pred))) * 100)


# You must provide a starting point at which to initialize
# the parameter search space

projects = os.listdir('D:\Dataset')
# projects = [f'D:/Dataset/{p}/{p}' for p in projects]
fig, axs = plt.subplots(2,2)
fig.set_size_inches(10,10)
dim = ['2d', '3d']
for i, data in enumerate(['2D', '3D']):
    # data.pop('microstructure054')
    data = datafull[f'validation_data{data}']
    err_exp_vf = np.array([data[n]['err_exp_vf'] for n in data.keys()])
    err_exp_sa = np.array([data[n]['err_exp_sa'] for n in data.keys()])
    pred_err_vf = np.array([data[n]['pred_err_vf'] for n in data.keys()])
    pred_err_sa = np.array([data[n]['pred_err_sa'] for n in data.keys()])

    vf_results = [err_exp_vf, pred_err_vf]
    sa_results = [err_exp_sa[pred_err_sa!=0], pred_err_sa[pred_err_sa!=0]]
    
    for j, (met, res) in enumerate(zip(['Volume fraction', 'Surface area'], [vf_results, sa_results])):
        ax = axs[i,j]
        ax.plot(np.arange(20), np.arange(20))
        beta_init = np.array([1, 0])
        result = minimize(objective_function, beta_init, args=(res[0], res[1]),
                  method='BFGS', options={'maxiter': 50000})
        y_data = result.x[0]*res[1] + result.x[1]
        # print(result.x[0], result.x[1])
        # y_data = res[1]
        ax.scatter(res[0], y_data, s=7, label='Predictions')
        x = np.arange(20)
        ax.plot(x, x, label = 'Ideal fit')
        
        ax.set_xlabel('Error from statistical analysis')
        ax.set_ylabel('Error from tpc analysis')
        ax.set_title(f'{met} {dim[i]}')
        ax.set_aspect(1)
        ax.set_yticks([0, 5, 10, 15, 20])
        errs = (y_data-res[0])/y_data
        errs = np.sort(errs)
        err = np.std((y_data-res[0])/y_data)*1.96/2
        print(err)
        ax.plot(x - x*err, x, c='black', ls='--', linewidth=1)
        ax.plot(x ,x -x*err, label = f'95% confidence ', c='black', ls='--', linewidth=1)
        # ax.plot(np.arange(20)/err, np.arange(20))



        # print(f'Unfitted {met} error {dim[i]}: {abs((res[0] - res[1])/res[0]).mean()}')
        # print(f'Fitted {met} error {dim[i]}: {abs((res[0] - y_data)/res[0]).mean()}')
axs[0,0].legend()
fig.savefig('fig3.pdf', format='pdf')         
# fig2.savefig('fig4.pdf', format='pdf')         




