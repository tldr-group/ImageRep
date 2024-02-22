import json
import numpy as np
import time
from scipy.stats import norm
import matplotlib.pyplot as plt

'''
File: prediction_error.py

Description: This script is used to analyse the prediction error of the integral range 
of the microstructures from the microlib dataset. 
'''

def error_by_size_estimation(dim, run_idx=0):
    data_dim, micro_names, slope_to_fit = data_micros_and_slope(dim, run_idx)
    edge_lengths_pred = data_dim['edge_lengths_pred']
    stds = []
    for edge_length in edge_lengths_pred:
        ir_results, err_results = comparison_results(data_dim, micro_names, str(edge_length))

        slope = slope_to_fit[dim]["Integral Range"]
        y_data = slope*ir_results[1] 
        
        errs = (y_data-ir_results[0])/y_data 
        errs = np.sort(errs)  # easier to see the distribution of errors

        std = np.std(errs) 
        stds.append(std)
        z = norm.interval(0.9)[1]
        err = std*z
        print(f'Integral Range {dim} {edge_length} std = {np.round(std,4)}')
        print(f'mean = {np.mean(errs)}')
        print(f'mape = {np.mean(np.abs(errs))}')
        print(f'error = {err}')
    return stds

def data_micros_and_slope(dim, run_idx=0):
    with open("microlib_statistics.json", "r") as fp:
        datafull = json.load(fp)

    with open("micro_names.json", "r") as fp:
        micro_names = json.load(fp)

    run_data = datafull[f'run_{run_idx}']
    run_data = run_data[f'data_gen_{dim}']
    
    
    # Slope for tpc to stat.analysis error fit, that have 0 mean: 
    slope_to_fit = {'2D': {'Integral Range': 1},
                    '3D': {'Integral Range': 0.95}}
    
    return run_data, micro_names, slope_to_fit

def comparison_results(data_dim, micro_names, edge_length):
    
    fit_ir_vf = np.array([data_dim[n]['fit_ir_vf'] for n in micro_names])
    fit_err_vf = np.array([data_dim[n]['fit_err_vf'][edge_length] for n in micro_names])
    pred_ir_vf = np.array([data_dim[n]['pred_ir_vf'][edge_length] for n in micro_names])
    pred_err_vf = np.array([data_dim[n]['pred_err_vf'][edge_length] for n in micro_names])
    
    ir_results = [fit_ir_vf, pred_ir_vf]
    err_results = [fit_err_vf, pred_err_vf]
    return ir_results, err_results

def plot_error_by_size(errs, edge_lengths_pred, dim):
    plt.plot(edge_lengths_pred, errs)
    plt.xlabel('Edge Length')
    plt.ylabel('Error')
    plt.title(f'Error by Edge Length {dim}')
    plt.savefig(f'error_by_size_{dim}.png')

if __name__ == '__main__':
    stds = []
    for i in range(5):
        stds.append(error_by_size_estimation('2D', i))
    stds = np.array(stds).sum(axis=0)/5
    run_data, _, _ = data_micros_and_slope('2D')
    edge_lengths_pred = run_data['edge_lengths_pred']
    plot_error_by_size(stds, edge_lengths_pred, '2D')
