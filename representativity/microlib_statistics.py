import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import time
from scipy.stats import norm


'''
File: microlib_statistics.py

Description: This script is used to generate the statistics for the representativity of 
microstructures from the microlib dataset. 
The statistics generated are then saved in a json file, to be analyzed and used for 
representativity prediction of a new microstructure or micrograph.
'''

def insert_v_names_in_run_data(run_data, mode, n, v_names):
    '''
    This function is used to insert the names of the variables that will be used to store the statistics'''
    if n not in run_data[f'data_gen_{mode}']:
            run_data[f'data_gen_{mode}'][n] = {}
            for v_name in v_names:
                if v_name not in run_data[f'data_gen_{mode}'][n]:
                    run_data[f'data_gen_{mode}'][n][v_name] = {}

def run_ir_prediction(netG, n, mode, imsize, run_data, conf):
    '''
    This function is used to predict the integral range of the microstructure.'''
    lf = imsize//32 + 2  # the size of G's input
    single_img = util.generate_image(netG, lf=lf, threed=mode=='3D', reps=1)
    if single_img.any():
        single_img = single_img.cpu()[0]
        pred_err_vf, _, pred_ir_vf = util.make_error_prediction(single_img, conf=conf, model_error=False, correction=False, mxtpc=100)
        print(f'pred ir {imsize} = {np.round(pred_ir_vf, 3)}')
        run_data[f'data_gen_{mode}'][n]['pred_ir_vf'][str(imsize)] = pred_ir_vf
        run_data[f'data_gen_{mode}'][n]['pred_err_vf'][str(imsize)] = pred_err_vf

def run_statistical_fit_analysis(netG, n, mode, edge_lengths_fit, run_data):
    
    imsize = 448 if mode=='3D' else 1600
    lf = imsize//32 + 2  # the size of G's input
    many_imgs = util.generate_image(netG, lf=lf, threed=mode=='3D', reps=50)
    vf = torch.mean(many_imgs).cpu().item()
    print(f'{n} vf = {vf}')
    run_data[f'data_gen_{mode}'][n]['vf'] = vf
    fit_ir_vf = util.stat_analysis_error(many_imgs, vf, edge_lengths_fit)
    run_data[f'data_gen_{mode}'][n]['fit_ir_vf'] = fit_ir_vf
    return fit_ir_vf, vf


def compare_to_prediction(n, mode, edge_lengths_pred, fit_ir_vf, vf, run_data, conf):

    n_dims = 2 if mode=='2D' else 3
    img_sizes = [(l,)*n_dims for l in edge_lengths_pred]
    fit_errs_vf = util.bernouli(vf, util.ns_from_dims(img_sizes, fit_ir_vf), conf=conf)
    for i in range(len(edge_lengths_pred)):
        imsize = edge_lengths_pred[i]
        run_data[f'data_gen_{mode}'][n]['fit_err_vf'][imsize] = fit_errs_vf[i]
    pred_errs = [run_data[f'data_gen_{mode}'][n]['pred_err_vf'][str(imsize)] for imsize in edge_lengths_pred]
    err_dict = {edge_lengths_pred[i]: (fit_errs_vf[i], pred_errs[i]) for i in range(len(edge_lengths_pred))}
    print(f'errors = {err_dict}')

def json_preprocessing(run_number=0):
    '''
    This function is used to load the data from the microlib dataset, and to prepare the json file
    '''

    # Load the statistics file
    with open("microlib_statistics.json", "r") as fp:
        all_data = json.load(fp)

    if f'run_{run_number}' not in all_data:
        all_data[f'run_{run_number}'] = {}
    run_data = all_data[f'run_{run_number}']

    # Dataset path and list of subfolders
    with open("micro_names.json", "r") as fp:
        micro_names = json.load(fp)
    micros = [f'/home/amir/microlibDataset/{p}/{p}' for p in micro_names]
    # Load placeholder generator
    netG = util.load_generator(micros[0])

    v_names = ['vf', 'pred_ir_vf', 'pred_err_vf', 'fit_err_vf']

    modes = ['2D', '3D']
    for mode in modes:
        if f'data_gen_{mode}' not in run_data:
            run_data[f'data_gen_{mode}'] = {}

    cur_modes = ['3D']  # change to ['2D'] or ['3D'] to only do one mode
    run_prediction, run_statistical_fit = True, False

    # Edge lengths for the experimental statistical analysis:
    run_data['data_gen_2D']['edge_lengths_fit'] = list(range(500, 1000, 20))
    run_data['data_gen_3D']['edge_lengths_fit'] = list(range(120, 400, 20))

    # Edge lengths for the predicted integral range:
    run_data['data_gen_2D']['edge_lengths_pred'] = list(range(8*32, 65*32, 4*32))
    run_data['data_gen_3D']['edge_lengths_pred'] = list(range(4*32, 19*32, 1*32))
    return all_data, run_data, micros, netG, cur_modes, v_names, run_prediction, run_statistical_fit

def run_microlib_statistics(run_number=0):
    '''
    This function is used to run the statistical analysis on the microlib dataset. 
    It will generate the statistics for each microstructure in the dataset, and save it in a json file.'''

    all_data, run_data, micros, netG, cur_modes, v_names, run_p, run_s = json_preprocessing(run_number)

    total_time_0 = time.time()
    # run the statistical analysis on the microlib dataset
    for j, p in enumerate(micros):

        try:
            netG.load_state_dict(torch.load(p + "_Gen.pt"))
        except:  # if the image is greayscale it's excepting because there's only 1 channel
            continue
        
        t_micro = time.time()
        for mode in cur_modes:

            print(f'{mode} mode')

            edge_lengths_fit = run_data[f'data_gen_{mode}']['edge_lengths_fit']
            edge_lengths_pred = run_data[f'data_gen_{mode}']['edge_lengths_pred']
            
            n = p.split('/')[-1]
            insert_v_names_in_run_data(run_data, mode, n, v_names)  # insert var names in run_data
            
            conf = 0.95  # confidence level for the prediction error

            if run_p:
                print(f'{n} starting prediction')
                for imsize in edge_lengths_pred:
                    run_ir_prediction(netG, n, mode, imsize, run_data, conf)

            if run_s:
                print(f'{n} starting statistical fit')
                fit_ir_vf, vf = run_statistical_fit_analysis(netG, n, mode, edge_lengths_fit, run_data)
                compare_to_prediction(n, mode, edge_lengths_pred, fit_ir_vf, vf, run_data, conf)
            
            print()
            with open(f"microlib_statistics.json", "w") as fp:
                json.dump(all_data, fp) 

        print(f'time for one micro {cur_modes} = {np.round((time.time()-t_micro)/60, 2)} minutes')

        print(f'{j+1}/{len(micros)} microstructures done')
    print(f'total time = {np.round((time.time()-total_time_0)/60, 2)} minutes')


if __name__ == '__main__':
    for i in range(5):
        run_microlib_statistics(5)
