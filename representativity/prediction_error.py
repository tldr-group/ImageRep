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

def error_by_size_estimation(dim, run_number=0):
    data_dim, micro_names, slope_to_fit = data_micros_and_slope(dim)
    edge_lengths_pred = data_dim['edge_lengths_pred']
    stds = []
    for edge_length in edge_lengths_pred:
        ir_results, err_results = comparison_results(data_dim, micro_names, str(edge_length), run_number)

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

def data_micros_and_slope(dim):
    with open("microlib_statistics_final.json", "r") as fp:
        datafull = json.load(fp)

    with open("micro_names.json", "r") as fp:
        micro_names = json.load(fp)

    dim_data = datafull[f'data_gen_{dim}']
    
    # Slope for tpc to stat.analysis error fit, that have 0 mean: 
    slope_to_fit = {'2D': {'Integral Range': 1},
                    '3D': {'Integral Range': 0.95}}
    
    return dim_data, micro_names, slope_to_fit

def comparison_results(data_dim, micro_names, edge_length, run_number=0):
    micros_data = [data_dim[n] for n in micro_names]
    fit_ir_vf = np.array([m_data['fit_ir_vf'] for m_data in micros_data])
    fit_err_vf = np.array([m_data['fit_err_vf'][edge_length] for m_data in micros_data])
    pred_ir_vf = np.array([m_data[f'run_{run_number}']['pred_ir_vf'][edge_length] for m_data in micros_data])
    pred_err_vf = np.array([m_data[f'run_{run_number}']['pred_err_vf'][edge_length] for m_data in micros_data])
    
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
    num_runs = 10
    for i in range(num_runs):
        stds.append(error_by_size_estimation('2D', i))
    stds = np.array(stds).sum(axis=0)/num_runs
    run_data, _, _ = data_micros_and_slope('2D')
    edge_lengths_pred = run_data['edge_lengths_pred']
    plot_error_by_size(stds, edge_lengths_pred, '2D')
