import json
import numpy as np
import time
from representativity import util
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from functools import partial

'''
File: prediction_error.py

Description: This script is used to analyse the prediction error of the integral range 
of the microstructures from the microlib dataset. 
'''

def error_by_size_estimation(dim, run_number=0, std_not_cls=True):
    data_dim, micro_names = data_micros(dim)
    edge_lengths_pred = data_dim['edge_lengths_pred']
    stds = []
    for edge_length in edge_lengths_pred:
        _, _, err, std, pfs = comparison_results(data_dim, micro_names, dim,
                                          str(edge_length), run_number, std_not_cls=std_not_cls)
        stds.append(std)
    return stds

def data_micros(dim):
    with open("microlib_statistics_periodic.json", "r") as fp:
        datafull = json.load(fp)

    with open("micro_names.json", "r") as fp:
        micro_names = json.load(fp)

    dim_data = datafull[f'data_gen_{dim}']
    
    
    return dim_data, micro_names

def comparison_results(data_dim, micro_names, dim, edge_length, run_number, std_not_cls=False):
    micros_data = [data_dim[n] for n in micro_names]
    pfs = np.array([m_data['vf'] for m_data in micros_data])
    fit_data_all = np.array([m_data['fit_ir_vf'] for m_data in micros_data])
    fit_err_pf = np.array([m_data['fit_err_vf'][edge_length] for m_data in micros_data])
    pred_data_all = np.array([m_data[f'run_{run_number}']['pred_ir_vf'][edge_length] for m_data in micros_data])
    # pred_ir_oi_pf = np.array([m_data[f'run_{run_number}']['pred_ir_one_im_fit_vf'][edge_length] for m_data in micros_data])

    pred_err_pf = np.array([m_data[f'run_{run_number}']['pred_err_vf'][edge_length] for m_data in micros_data])
    if std_not_cls:
        pfs = np.array(pfs)
        pfs_one_minus_pfs = pfs*(1-pfs)  
        dim_int = int(dim[0])
        edge_length = int(edge_length)
        pred_data_all = ((pred_data_all/edge_length)**dim_int*pfs_one_minus_pfs)**0.5
        fit_data_all = ((fit_data_all/edge_length)**dim_int*pfs_one_minus_pfs)**0.5
    # ir_results = [fit_ir_pf, pred_ir_pf, pred_ir_oi_pf]
    ir_results = [fit_data_all, pred_data_all]
    err_results = [fit_err_pf, pred_err_pf]
    pred_data = ir_results[1] 
    fit_data = ir_results[0]

    errs = (pred_data-fit_data)/pred_data  # percentage error of the prediction
    errs = np.sort(errs)  # easier to see the distribution of errors
    std = np.std(errs) 
    z = norm.interval(0.9)[1]
    err = std*z
    # print(f'Integral Range {dim} {edge_length} std = {np.round(std,4)}')
    # print(f'mean = {np.mean(errs)}')
    # print(f'mape = {np.mean(np.abs(errs))}')
    # print(f'error = {err}')
    return pred_data, fit_data, err, std, pfs

def pred_vs_fit_all_data(dim, edge_length, num_runs=5, std_not_cls=True):
    pred_data_all = []
    pred_data_oi_all = []
    fit_data_all = []
    stds = []
    pfs_all = []
    for i in range(num_runs):
        data_micro = data_micros(dim)
        pred_data, fit_data, _, std, pfs = comparison_results(*data_micro, dim, edge_length, i, std_not_cls=std_not_cls)
        pred_data_all.append(pred_data)
        # pred_data_oi_all.append(pred_oi_data)
        fit_data_all.append(fit_data)
        stds.append(std)
        pfs_all.append(pfs) 
    pred_data_all = np.concatenate(pred_data_all)
    # pred_data_oi_all = np.concatenate(pred_data_oi_all)
    fit_data_all = np.concatenate(fit_data_all)
    pfs_all = np.concatenate(pfs_all)
    std = np.array(stds).sum(axis=0)/num_runs
    return pred_data_all, fit_data_all, std, pfs_all

def plot_pred_vs_fit(dim, edge_length, num_runs=5, std_not_cls=True):
    pred_data_all, fit_data_all, _, pfs = pred_vs_fit_all_data(dim, edge_length, num_runs, std_not_cls=std_not_cls)
    pred_data_all, fit_data_all = np.array(pred_data_all), np.array(fit_data_all)
    
    errs = (fit_data_all-pred_data_all)/pred_data_all  # percentage error of the prediction
    errs = np.sort(errs)  # easier to see the distribution of errors
    std = np.std(errs) 
    z = norm.interval(0.95)[1]
    err = std*z
    print(f'Integral Range {dim} {edge_length} std = {np.round(std,4)}')
    print(f'mean = {np.mean(errs)}')
    print(f'mape = {np.mean(np.abs(errs))}')
    print(f'error = {err}')
    print(f'total shift = {np.round(np.mean(errs) + err,5)}')
    print(f'error in shift = {100*np.round(np.mean(errs)/(np.mean(errs) + err),5)}%')
    
    plt.scatter(fit_data_all, pred_data_all, s=0.2)
    max_val = np.max([np.max(fit_data_all),np.max(pred_data_all)])
    plt.plot(np.arange(0, max_val, 0.001), np.arange(0, max_val, 0.001), c='k')
    plt.xlabel('Fit Data')
    plt.ylabel('Pred Data')
    plt.title(f'Prediction vs. Fit Data {dim}')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f'pred_vs_fit_all_runs_periodic_cut/pred_vs_fit_{dim}_{edge_length}.png')
    plt.show()
    plt.close()

def std_error_by_size(dim, edge_lengths, num_runs=5, start_idx=0, std_not_cls=True):
    stds = []
    for i in range(num_runs):
        stds.append(error_by_size_estimation(dim, i, std_not_cls=std_not_cls))
    stds = np.array(stds).sum(axis=0)/num_runs
    n_voxels = np.array([edge**int(dim[0]) for edge in edge_lengths])
    stds, n_voxels = stds[start_idx:], n_voxels[start_idx:]
    popt, pcov = curve_fit(partial(util.fit_to_errs_function, dim), n_voxels, stds)
    # img_sizes = [(l,)*2 for l in edge_lengths]
    # pfs, irs = [0.1, 0.2, 0.4], [40, 40, 40]
    # for i in range(len(pfs)):
        # erros_inherent = util.bernouli(pfs[i], util.ns_from_dims(img_sizes, irs[i]),conf=0.95)
        # plt.plot(edge_lengths, erros_inherent, label=f'Inherent error IR = {irs[i]}, pf = {pfs[i]}')
    print(f'popt: {popt}')
    prediction_error = util.fit_to_errs_function(dim, n_voxels, *popt)*100
    return prediction_error, stds

def plot_std_error_by_size(dim, edge_lengths, start_idx=0, num_runs=5):
    pred_error, stds = std_error_by_size(dim, edge_lengths, start_idx=start_idx, num_runs=5)
    edge_lengths = edge_lengths[start_idx:]
    plt.scatter(edge_lengths, stds*100, label='Prediction error std')
    plt.plot(edge_lengths, pred_error, label='Prediction error fit')
    plt.xlabel('Edge Length')
    plt.ylabel('MPE std (%)')
    plt.title(f'Error by Edge Length {dim}')
    plt.legend()
    plt.savefig(f'error_by_size_{dim}_pf.png')
    plt.show()
    plt.close()


def find_optimal_slope(pred_data, fit_data):
    def mape(slope):
        pred_data_scaled = slope * pred_data
        return np.mean(np.abs(pred_data_scaled - fit_data) / pred_data_scaled)
    
    result = minimize(mape, x0=1, bounds=[(0, 2)])
    optimal_slope = result.x[0]
    return optimal_slope

def optimal_slopes(dim, num_runs=5):
    data_micros = data_micros(dim)
    edge_lengths_pred = data_micros[0]['edge_lengths_pred']
    slopes = []
    stds = []
    for edge_length in edge_lengths_pred:
        pred_data, fit_data, std, pfs = pred_vs_fit_all_data(dim, str(edge_length), num_runs)
        stds.append(std)
        optimal_slope = find_optimal_slope(pred_data, fit_data)
        slopes.append(optimal_slope)
    
    return edge_lengths_pred, slopes, stds

if __name__ == '__main__':
    dim = '3D'
    num_runs_cur = 10
    run_data, _ = data_micros(dim)
    edge_lengths_pred = run_data['edge_lengths_pred']
    plot_std_error_by_size(dim, edge_lengths_pred, start_idx=2, num_runs=num_runs_cur)
    
    run_data, _ = data_micros(dim)
    edge_lengths_pred = run_data['edge_lengths_pred']
    for edge_length in edge_lengths_pred:
        plot_pred_vs_fit(dim, str(edge_length), num_runs=num_runs_cur)
    
    
    