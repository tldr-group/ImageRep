simport json
import numpy as np
from scipy.optimize import curve_fit
import sys
sys.path.append('../')
from representativity.old import util
import matplotlib.pyplot as plt
from matplotlib import animation 
from scipy.optimize import minimize
from scipy.stats import norm
import math
import os

print(os.getcwd())
with open("../microlib_statistics.json", "r") as fp:
    datafull = json.load(fp)

# Slope for tpc to stat.analysis error fit, that have 0 mean: 
slope_to_fit = {'2D': {'Integral Range': 1, 'Percentage Error': 1},
                '3D': {'Integral Range': 0.95, 'Percentage Error': 0.95**(1/3)}}

with open("../micro_names.json", "r") as fp:
    micro_names = json.load(fp)
projects = [f'/home/amir/microlibDataset/{p}/{p}' for p in micro_names]
fig, axs = plt.subplots(1,1)
fig.set_size_inches(8,8)
dims = ['2D', '3D']

def animate_with_data(edge_length, data_dim, dim):
    global micro_names
    end_idx = 77 if dim == '3D' else 78
    micro_names = micro_names[:end_idx]
    fit_ir_vf = np.array([data_dim[n]['fit_ir_vf'] for n in micro_names])
    fit_err_vf = np.array([data_dim[n]['fit_err_vf'][edge_length] for n in micro_names])
    pred_ir_vf = np.array([data_dim[n]['pred_ir_vf'][edge_length] for n in micro_names])
    pred_err_vf = np.array([data_dim[n]['pred_err_vf'][edge_length] for n in micro_names])
    
    ir_results = [fit_ir_vf, pred_ir_vf]
    err_results = [fit_err_vf, pred_err_vf]

    for j, (met, res) in enumerate(zip(['Integral Range'], [ir_results])):
        plt.cla()
        ax = axs
    
        slope = slope_to_fit[dim][met]
        y_data = slope*res[1] 
        
        ax.scatter(res[0], y_data, s=7, label='Predictions')
        
        ax.set_xlabel(f'IR from stat. analysis')
        ax.set_ylabel(f'IR from TPC')
        ax.set_title(f'{met} {dim} image length size: {edge_length}')
        
        errs = (res[0]-y_data)/y_data 
        
        max_val = int(max(fit_ir_vf))+10
    
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
        print(f'{met} {dim} {edge_length} std = {np.round(std,4)}')
        print(f'mean = {np.mean(errs)}')
        print(f'mape = {np.mean(np.abs(errs))}')
        print(f'error = {err}')
        
        ax.plot(np.arange(max_val), np.arange(max_val), c='black')
        ax.plot(x ,x/(1+err), c='black', ls='--', linewidth=1)
        fill_1 = ax.fill_between(x, np.ones(x.shape[0])*(max_val),x/(1+err), alpha=0.2, label = f'95% confidence range')
        ax.set_aspect('equal')
        
        # print(f'Unfitted {met} MAPE error {dims[i]}: {abs((res[0] - res[1])/res[0]).mean()}')
        # print(f'Fitted {met} MAPE error {dims[i]}: {abs((res[0] - y_data)/res[0]).mean()}')
    ax.legend()

    # fig.savefig('fig3.pdf', format='pdf')

def gif_plot(dim):
    data_dim = datafull[f'data_gen_{dim}']
    edge_lengths = data_dim["edge_lengths_pred"]

    def animate(edge_length):
        animate_with_data(str(edge_length), data_dim, dim)
        
    # animate(edge_lengths[-1])
    ani = animation.FuncAnimation(fig, animate, repeat=True, 
                                  frames=edge_lengths, interval=50)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=2)
    ani.save('error_as_a_function_of_size.gif', writer=writer)

    plt.show()


# def old_plot():
#     for i, data in enumerate(dims):
#         data_dim = datafull[f'data_gen_{data}']
#         err_exp_vf = np.array([data_dim[n]['exp_ir_vf'] for n in data_dim.keys()])
#         err_exp_sa = np.array([data_dim[n]['exp_err_vf'] for n in data_dim.keys()])
#         pred_err_vf = np.array([data_dim[n]['pred_ir_vf'] for n in data_dim.keys()])
#         pred_err_sa = np.array([data_dim[n]['pred_err_vf'] for n in data_dim.keys()])
#         data_2d = datafull[f'validation_data2D']
#         # dim_var = np.array([data_2d[n]['dim_variation'] if 'dim_variation' in data_2d[n] else 0 for n in data_2d.keys()])
#         # dim_var = dim_var/np.max(dim_var) if np.max(dim_var) > 0 else dim_var
#         # colors = np.array([(0,0,0) if n in periodic_micros else (1,0,0) for n in data.keys()])
#         # colors = np.array([(dim_var[i],0,0) for i in range(len(dim_var))])
#         vf_results = [err_exp_vf, pred_err_vf]
#         sa_results = [[err_exp_sa[0]], [pred_err_sa[0]]]
#         for idx in range(1, len(err_exp_sa)):
#             if not math.isnan(err_exp_sa[idx]) and not math.isnan(pred_err_sa[i]):
#                 sa_results[0].append(err_exp_sa[idx])
#                 sa_results[1].append(pred_err_sa[idx])
#         sa_results[0], sa_results[1] = np.array(sa_results[0]), np.array(sa_results[1])
#         # sa_results = [err_exp_sa[pred_err_sa!=math.isnan], pred_err_sa[pred_err_sa!=math.nan]]
#         for j, (met, res) in enumerate(zip(['Volume fraction', 'Surface area'], [vf_results, sa_results])):
#             ax = axs[i, j]
#             end_idx = 20
#             res[0], res[1] = res[0][:end_idx], res[1][:end_idx]
#             # colors = colors[:end_idx]
        
#             slope = slope_to_fit[dims[i]][met]
#             y_data = slope*res[1] 
            
#             without_last_outlier = np.logical_and(y_data > 0, res[0] > 0)
#             ax.scatter(res[0][without_last_outlier], y_data[without_last_outlier], s=7, label='Predictions')
            
#             ax.set_xlabel(f'IR from stat. analysis')
#             ax.set_ylabel(f'IR from TPC')
#             ax.set_title(f'{met} {dims[i]}')
            
#             errs = (res[0]-y_data)/y_data 
            
#             max_val = int(np.max([np.max(y_data[without_last_outlier]),np.max(res[0][without_last_outlier])]))+2
            
#             x = np.arange(max_val+1)
#             ax.plot(x, x, label = 'Ideal error prediction', color='black')
#             ax.set_yticks(np.arange(0, max_val, 5))
#             ax.set_xticks(np.arange(0, max_val, 5))
#             ax.set_xlim(right=67)
#             ax.set_ylim(top=67)
#             errs = np.sort(errs) 
#             std = np.std((y_data-res[0])/y_data) 
            
#             z = norm.interval(0.9)[1]
#             err = std*z
#             print(f'{met} {dims[i]} std = {std}')
#             print(f'mean = {np.mean(errs)}')
#             print(f'mape = {np.mean(np.abs(errs))}')
#             print(f'error = {err}')
            
#             ax.plot(np.arange(max_val), np.arange(max_val), c='black')
#             ax.plot(x ,x/(1+err), c='black', ls='--', linewidth=1)
#             fill_1 = ax.fill_between(x, np.ones(x.shape[0])*(max_val),x/(1+err), alpha=0.2, label = f'95% confidence range')
#             ax.set_aspect('equal')
            
#             # print(f'Unfitted {met} MAPE error {dims[i]}: {abs((res[0] - res[1])/res[0]).mean()}')
#             # print(f'Fitted {met} MAPE error {dims[i]}: {abs((res[0] - y_data)/res[0]).mean()}')
#     axs[0, 0].legend()  # TODO add text to the legend indicating the MAPE
#     fig.savefig('fig3.pdf', format='pdf')         

if __name__ == '__main__':
    gif_plot('2D')

