import json
import numpy as np
import time
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

def error_by_size_estimation(dim, run_number=0, after_slope_calc=True):
    data_dim, micro_names, slope = data_micros_and_slope(dim, after_slope_calc)
    edge_lengths_pred = data_dim['edge_lengths_pred']
    stds = []
    for edge_length in edge_lengths_pred:
        _, _, _, std = comparison_results(data_dim, micro_names, slope, dim,
                                          str(edge_length), run_number, after_slope_calc)
        stds.append(std)
    return stds

def data_micros_and_slope(dim, after_slope_calc=True):
    with open("microlib_statistics_final.json", "r") as fp:
        datafull = json.load(fp)

    with open("micro_names.json", "r") as fp:
        micro_names = json.load(fp)

    dim_data = datafull[f'data_gen_{dim}']
    
    # Slope for tpc to stat.analysis error fit, that have 0 mean: 
    if after_slope_calc:
        slope_to_fit = {'2D': {'Integral Range': [28295.03239128, 10896.29626247]},
                        '3D': {'Integral Range': 0.95}}
    else:
        slope_to_fit = {'2D': {'Integral Range': 1},
                        '3D': {'Integral Range': 0.95}}
    
    return dim_data, micro_names, slope_to_fit

def comparison_results(data_dim, micro_names, slope, dim, edge_length, run_number, after_slope_calc=True):
    micros_data = [data_dim[n] for n in micro_names]
    fit_ir_vf = np.array([m_data['fit_ir_vf'] for m_data in micros_data])
    fit_err_vf = np.array([m_data['fit_err_vf'][edge_length] for m_data in micros_data])
    pred_ir_vf = np.array([m_data[f'run_{run_number}']['pred_ir_vf'][edge_length] for m_data in micros_data])
    pred_err_vf = np.array([m_data[f'run_{run_number}']['pred_err_vf'][edge_length] for m_data in micros_data])
    
    ir_results = [fit_ir_vf, pred_ir_vf]
    err_results = [fit_err_vf, pred_err_vf]

    if after_slope_calc:
        slope_coefficients = slope[dim]["Integral Range"]
        slope = fit_to_slope_function(dim, [int(edge_length)], *slope_coefficients)
    else:
        slope = slope[dim]["Integral Range"]
    pred_data = slope*ir_results[1] 
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
    return pred_data, fit_data, errs, std

def pred_vs_fit_all_data(dim, edge_length, num_runs=5, after_slope_calc=True):
    pred_data_all = []
    fit_data_all = []
    stds = []
    for i in range(num_runs):
        pred_data, fit_data, _, std = comparison_results(*data_micros_and_slope(dim, after_slope_calc), dim, edge_length, i, after_slope_calc)
        pred_data_all.append(pred_data)
        fit_data_all.append(fit_data)
        stds.append(std)
    pred_data_all = np.concatenate(pred_data_all)
    fit_data_all = np.concatenate(fit_data_all)
    std = np.array(stds).sum(axis=0)/num_runs
    return pred_data_all, fit_data_all, std

def plot_pred_vs_fit(dim, edge_length, num_runs=5, after_slope_calc=True):
    pred_data_all, fit_data_all, _ = pred_vs_fit_all_data(dim, edge_length, num_runs, after_slope_calc)
    
    errs = (pred_data_all-fit_data_all)/pred_data_all  # percentage error of the prediction
    errs = np.sort(errs)  # easier to see the distribution of errors
    std = np.std(errs) 
    z = norm.interval(0.9)[1]
    err = std*z
    print(f'Integral Range {dim} {edge_length} std = {np.round(std,4)}')
    print(f'mean = {np.mean(errs)}')
    print(f'mape = {np.mean(np.abs(errs))}')
    print(f'error = {err}')
    
    plt.scatter(fit_data_all, pred_data_all, s=0.2)
    plt.xlabel('Fit Data')
    plt.ylabel('Pred Data')
    plt.title(f'Prediction vs. Fit Data {dim}')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f'pred_vs_fit_{dim}.png')
    plt.show()
    plt.close()

def plot_std_error_by_size(dim, edge_lengths, num_runs=5, after_slope_calc=True):
    stds = []
    for i in range(num_runs):
        stds.append(error_by_size_estimation(dim, i, after_slope_calc))
    stds = np.array(stds).sum(axis=0)/num_runs
    popt, pcov = curve_fit(partial(fit_to_errs_function, dim), edge_lengths, stds)
    print(f'popt: {popt}')
    plt.scatter(edge_lengths_pred, stds)
    plt.plot(edge_lengths_pred, fit_to_errs_function(dim, edge_lengths_pred, *popt), label='Fit')
    plt.xlabel('Edge Length')
    plt.ylabel('Error')
    plt.title(f'Error by Edge Length {dim}')
    plt.savefig(f'error_by_size_{dim}.png')
    plt.show()


def find_optimal_slope(pred_data, fit_data):
    def mape(slope):
        pred_data_scaled = slope * pred_data
        return np.mean(np.abs(pred_data_scaled - fit_data) / pred_data_scaled)
    
    result = minimize(mape, x0=1, bounds=[(0, 2)])
    optimal_slope = result.x[0]
    return optimal_slope

def optimal_slopes(dim, num_runs=5, after_slope_calc=True):
    data_micros_slope = data_micros_and_slope(dim, after_slope_calc)
    edge_lengths_pred = data_micros_slope[0]['edge_lengths_pred']
    slopes = []
    stds = []
    for edge_length in edge_lengths_pred:
        pred_data, fit_data, std = pred_vs_fit_all_data(dim, str(edge_length), num_runs, after_slope_calc)
        stds.append(std)
        optimal_slope = find_optimal_slope(pred_data, fit_data)
        slopes.append(optimal_slope)
    
    return edge_lengths_pred, slopes, stds

def fit_to_errs_function(dim, edge_lengths, a):
    edge_lengths = np.array(edge_lengths)
    power = 2 if dim == '2D' else 3
    return a / edge_lengths 

def fit_to_slope_function(dim, edge_lengths, a, b):
    edge_lengths = np.array(edge_lengths)
    if dim == '2D':
        size_threshold_1 = 500
        size_threshold_2 = 2000
        power = 2
    elif dim == '3D':
        size_threshold_1 = 250
        size_threshold_2 = 350
        power = 3
    small_edge_lengths = edge_lengths[edge_lengths < size_threshold_1]
    medium_edge_lengths = edge_lengths[(edge_lengths >= size_threshold_1) & (edge_lengths < size_threshold_2)]
    large_edge_lengths = edge_lengths[edge_lengths >= size_threshold_2]
    small_res = 1 + a / small_edge_lengths**power
    coeff = (medium_edge_lengths-size_threshold_1)/(size_threshold_2-size_threshold_1)
    medium_res = 1 + (1-coeff)*a/medium_edge_lengths**power + coeff*b/medium_edge_lengths**power
    large_res = 1 + b / large_edge_lengths**power
    return np.concatenate([small_res, medium_res, large_res])

def fit_to_slope(edge_lengths, slopes, dim='2D'):
    popt, pcov = curve_fit(partial(fit_to_slope_function, dim), edge_lengths, slopes)
    return popt

def plot_optimal_slopes(dim, num_runs=5, after_slope_calc=False):
    edge_lengths_pred, optimal_slopes_list, stds = optimal_slopes(dim, num_runs, after_slope_calc)
    optimal_slopes_list = np.array(optimal_slopes_list)
    popt = fit_to_slope(edge_lengths_pred, optimal_slopes_list)
    print(f'edge_lengths: {edge_lengths_pred}')
    print(f'slopes: {optimal_slopes_list}')
    print(f'Optimal Slope Fit: {popt}/n^2')
    print(f'Slopes fit: {fit_to_slope_function(dim, edge_lengths_pred, *popt)}')
    plt.scatter(edge_lengths_pred, optimal_slopes_list, label='Optimal Slopes')
    plt.plot(edge_lengths_pred, fit_to_slope_function(dim, edge_lengths_pred, *popt), label='Fit')
    plt.xlabel('Edge Length')
    plt.ylabel('Optimal Slope')
    plt.title(f'Optimal Slope by Edge Length {dim}')
    plt.legend()
    plt.savefig(f'optimal_slope_{dim}.png')
    plt.show()

if __name__ == '__main__':
    dim = '2D'
    after_slope_calc=True
    run_data, _, _ = data_micros_and_slope(dim, after_slope_calc)
    edge_lengths_pred = run_data['edge_lengths_pred']
    plot_std_error_by_size(dim, edge_lengths_pred, num_runs=10, after_slope_calc=after_slope_calc)
    
    # run_data, _, _ = data_micros_and_slope(dim, after_slope_calc)
    # edge_lengths_pred = run_data['edge_lengths_pred']
    # for edge_length in edge_lengths_pred:
    #     plot_pred_vs_fit(dim, str(edge_length), num_runs=10, after_slope_calc=after_slope_calc)

    # plot_optimal_slopes(dim, num_runs=10, after_slope_calc=False)
    
    
    