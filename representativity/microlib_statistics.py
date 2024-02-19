import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import time

'''
File: microlib_statistics.py

Description: This script is used to generate the statistics for the representativity of 
microstructures from the microlib dataset. 
The statistics generated are then saved in a json file, to be analyzed and used for 
representativity prediction of a new microstructure or micrograph.
'''

def insert_v_names_in_datafin(datafin, mode, n, v_names):
    '''
    This function is used to insert the names of the variables that will be used to store the statistics'''
    if n not in datafin[f'data_gen_{mode}']:
            datafin[f'data_gen_{mode}'][n] = {}
            for v_name in v_names:
                if v_name not in datafin[f'data_gen_{mode}'][n]:
                    datafin[f'data_gen_{mode}'][n][v_name] = {}

def run_ir_prediction(netG, n, mode, imsize, datafin, conf):
    '''
    This function is used to predict the integral range of the microstructure.'''
    lf = imsize//32 + 2  # the size of G's input
    single_img = util.generate_image(netG, lf=lf, threed=mode=='3D', reps=1)
    if single_img.any():
        single_img = single_img.cpu()[0]
        pred_err_vf, _, pred_ir_vf = util.make_error_prediction(single_img, conf=conf, model_error=False, correction=False, mxtpc=100)
        print(f'pred ir {imsize} = {pred_ir_vf}')
        datafin[f'data_gen_{mode}'][n]['pred_ir_vf'][str(imsize)] = pred_ir_vf
        datafin[f'data_gen_{mode}'][n]['pred_err_vf'][str(imsize)] = pred_err_vf

def run_statistical_fit_analysis(netG, n, mode, edge_lengths_fit, datafin):
    
    imsize = 448 if mode=='3D' else 1600
    lf = imsize//32 + 2  # the size of G's input
    many_imgs = util.generate_image(netG, lf=lf, threed=mode=='3D', reps=100)
    vf = torch.mean(many_imgs).cpu().item()
    print(f'{n} vf = {vf}')
    datafin[f'data_gen_{mode}'][n]['vf'] = vf
    fit_ir_vf = util.stat_analysis_error(many_imgs, vf, edge_lengths_fit)
    datafin[f'data_gen_{mode}'][n]['fit_ir_vf'] = fit_ir_vf
    return fit_ir_vf, vf


def compare_to_prediction(n, mode, edge_lengths_pred, fit_ir_vf, vf, datafin, conf):

    n_dims = 2 if mode=='2D' else 3
    img_sizes = [(l,)*n_dims for l in edge_lengths_pred]
    fit_errs_vf = util.bernouli(vf, util.ns_from_dims(img_sizes, fit_ir_vf), conf=conf)
    for i in range(len(edge_lengths_pred)):
        imsize = edge_lengths_pred[i]
        datafin[f'data_gen_{mode}'][n]['fit_err_vf'][imsize] = fit_errs_vf[i]
    pred_errs = [datafin[f'data_gen_{mode}'][n]['pred_err_vf'][str(imsize)] for imsize in edge_lengths_pred]
    err_dict = {edge_lengths_pred[i]: (fit_errs_vf[i], pred_errs[i]) for i in range(len(edge_lengths_pred))}
    print(f'errors = {err_dict}')

def json_preprocessing():
    '''
    This function is used to load the data from the microlib dataset, and to prepare the json file
    '''

    # Load the statistics file
    with open("microlib_statistics.json", "r") as fp:
        datafin = json.load(fp)
    # Dataset path and list of subfolders
    with open("micro_names.json", "r") as fp:
        micro_names = json.load(fp)
    micros = [f'/home/amir/microlibDataset/{p}/{p}' for p in micro_names]
    # Load placeholder generator
    netG = util.load_generator(micros[0])

    v_names = ['vf', 'pred_ir_vf', 'pred_err_vf', 'fit_err_vf']

    modes = ['2D', '3D']
    for mode in modes:
        if f'data_gen_{mode}' not in datafin:
            datafin[f'data_gen_{mode}'] = {}

    cur_modes = ['2D']  # change to ['2D'] or ['3D'] to only do one mode
    run_prediction, run_statistical_fit = True, True

    # Edge lengths for the experimental statistical analysis:
    datafin['data_gen_2D']['edge_lengths_fit'] = list(range(500, 1000, 20))
    datafin['data_gen_3D']['edge_lengths_fit'] = list(range(120, 400, 20))

    # Edge lengths for the predicted integral range:
    datafin['data_gen_2D']['edge_lengths_pred'] = list(range(8*32, 65*32, 4*32))
    datafin['data_gen_3D']['edge_lengths_pred'] = list(range(4*32, 19*32, 1*32))
    return datafin, micros, netG, cur_modes, v_names, run_prediction, run_statistical_fit

def run_microlib_statistics():
    '''
    This function is used to run the statistical analysis on the microlib dataset. 
    It will generate the statistics for each microstructure in the dataset, and save it in a json file.'''

    datafin, micros, netG, cur_modes, v_names, run_p, run_s = json_preprocessing()

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

            edge_lengths_fit = datafin[f'data_gen_{mode}']['edge_lengths_fit']
            edge_lengths_pred = datafin[f'data_gen_{mode}']['edge_lengths_pred']
            
            n = p.split('/')[-1]
            insert_v_names_in_datafin(datafin, mode, n, v_names)  # insert var names in datafin
            
            conf = 0.95  # confidence level for the prediction error

            if run_p:
                print(f'{n} starting prediction')
                for imsize in edge_lengths_pred:
                    run_ir_prediction(netG, n, mode, imsize, datafin, conf)

            if run_s:
                print(f'{n} starting statistical fit')
                fit_ir_vf, vf = run_statistical_fit_analysis(netG, n, mode, edge_lengths_fit, datafin)
                compare_to_prediction(n, mode, edge_lengths_pred, fit_ir_vf, vf, datafin, conf)
            
            print()
            with open(f"microlib_statistics.json", "w") as fp:
                json.dump(datafin, fp) 

        print(f'time for one micro {cur_modes} = {np.round((time.time()-t_micro)/60, 2)} minutes')

        print(f'{j+1}/{len(micros)} microstructures done')
    print(f'total time = {np.round((time.time()-total_time_0)/60, 2)} minutes')
    



if __name__ == '__main__':
    run_microlib_statistics()

# fit_err_vfs.append(fit_err_vf)
                # fit_ir_vfs.append(fit_ir_vf)
                # print(f'{j} pred error vf = {pred_err_vf}')
                # print(f'{j} experiment error vf = {err_fit_vf}')
                # print(f'% diff vf = {(err_fit_vf-pred_err_vf)/err_fit_vf}')
                # print()
                
                # print(f'{j} starting to sa')
                # sa_images = util.make_sas(img[0])
                # # Do calcs on single image
                # print(f'{j} starting to testing error')
                # sa = torch.mean(util.sa_map_from_sas(sa_images)).cpu().item()
                # print(f'{j} sa = {sa}')
                # sa_testimg = [sa_img[0, :l, :l].cpu() if mode=='2D' else sa_img[0, :l, :l, :l].cpu() for sa_img in sa_images]
                # pred_err_sa, _, tpc_sa_dist, tpc_sa, pred_ir_sa = util.make_error_prediction(sa_testimg, sa, model_error=False, correction=False)
                # compared_shape = [np.array(sa_testimg[0].size())]
                # err_fit_sa, fit_ir_sa = util.stat_analysis_error(util.sa_map_from_sas(sa_images), edge_lengths, img_dims, compared_shape, conf=0.95)
                # print(f'{j} pred error = {pred_err_sa}')
                # print(f'{j} experiment error = {err_fit_sa}')
                # print(f'% diff = {(err_fit_sa-pred_err_sa)/err_fit_sa}')

                # pred_err_sas.append(pred_err_sa)
                # pred_ir_sas.append(pred_ir_sa)
                # fit_err_sas.append(err_fit_sa)
                # fit_ir_sas.append(fit_ir_sa)

                # print(f'{j} starting real image stats')
                # Do stats on the full generated image
                
                # err_fit_vf = util.real_image_stats(img[0], edge_lengths, vf)[0]  
                # err_fit_sa = util.real_image_stats(util.sa_map_from_sas(sa_images), edge_lengths, sa)[0]  # TODO calc with new sa
                
                # tpc_vf = [list(tpc) for tpc in tpc_vf]
                # tpc_sa = [list(tpc) for tpc in tpc_sa]
                # data_val[n] = {'pred_err_vf': pred_err_vf.astype(np.float64),
                        #    'pred_err_sa':pred_err_sa.astype(np.float64),
                        #    'err_fit_vf':err_fit_vf.item(),
                        #    'err_fit_sa':err_fit_sa.item(),
                            # 'tpc_vf_dist':list(tpc_vf_dist),
                            # 'tpc_vf':tpc_vf,
                            # 'tpc_sa_dist':list(tpc_sa_dist),
                            # 'tpc_sa':tpc_sa
                            # }

# print(errs_ir_vf)
            # fit_ir_vf = np.mean(errs_ir_vf)
        #     print(f'mean ir = {fit_ir_vf}')
            # datafin[f'data_gen_{mode}'] = data_val
            # datafin[f'data_gen_{mode}'][n]['sa'] = sa
            # datafin[f'data_gen_{mode}'][n]['pred_err_sa'] = pred_err_sa.astype(np.float64)
            # datafin[f'data_gen_{mode}'][n]['pred_ir_sa'] = pred_ir_sa
            # datafin[f'data_gen_{mode}'][n]['err_fit_sa'] = err_fit_sa.item()
            # datafin[f'data_gen_{mode}'][n]['fit_ir_sa'] = fit_ir_sa

            # datafin[f'data_gen_{mode}'][n]['tpc_sa_dist'] = list(tpc_sa_dist)
            # datafin[f'data_gen_{mode}'][n]['tpc_sa'] = tpc_sa

# print(pred_ir_sas)
            # print(f'mean ir = {np.mean(pred_ir_sas)}')
            # datafin[f'data_gen_{mode}'][n]['pred_ir_sa'] = np.mean(pred_ir_sas)
            # if mode=='3D':
            #     datafin[f'data_gen_{mode}'][n]['dim_variation'] = 0
            # else:
            #     datafin[f'data_gen_{mode}'][n]['dim_variation'] = np.var(pred_ir_sas)/np.mean(pred_ir_sas)
            #     print(f'dim variation = {np.var(pred_ir_sas)/np.mean(pred_ir_sas)}')
            # datafin[f'data_gen_{mode}'][n]['pred_err_sa'] = np.mean(pred_err_sas).astype(np.float64)

            # print(fit_ir_vfs)
            # print(f'mean ir = {np.mean(fit_ir_vfs)}')
            # datafin[f'data_gen_{mode}'][n]['fit_ir_vf'] = np.mean(fit_ir_vfs)
            # datafin[f'data_gen_{mode}'][n]['fit_ir_vf'] = np.mean(fit_ir_vfs)
            # if mode=='3D':
            #     datafin[f'data_gen_{mode}'][n]['fit_dim_variation'] = 0
            # else:
            #     datafin[f'data_gen_{mode}'][n]['fit_dim_variation'] = np.var(fit_ir_vfs)/np.mean(fit_ir_vfs)
            #     print(f'dim variation = {np.var(fit_ir_vfs)/np.mean(fit_ir_vfs)}')
            # datafin[f'data_gen_{mode}'][n]['fit_err_vf'] = np.mean(fit_err_vfs).astype(np.float64)

            # print(fit_ir_sas)
            # print(f'mean ir = {np.mean(fit_ir_sas)}')
            # datafin[f'data_gen_{mode}'][n]['fit_ir_sa'] = np.mean(fit_ir_sas)
            # if mode=='3D':
            #     datafin[f'data_gen_{mode}'][n]['fit_dim_variation'] = 0
            # else:
            #     datafin[f'data_gen_{mode}'][n]['fit_dim_variation'] = np.var(fit_ir_sas)/np.mean(fit_ir_sas)
            #     print(f'dim variation = {np.var(fit_ir_sas)/np.mean(fit_ir_sas)}')
            # datafin[f'data_gen_{mode}'][n]['fit_err_sa'] = np.mean(fit_err_sas).astype(np.float64)

            # datafin[f'data_gen_{mode}'][n]['err_fit_vf'] = err_fit_vf.item()
            # datafin[f'data_gen_{mode}'][n]['fit_ir_vf'] = fit_ir_vf
            # datafin[f'data_gen_{mode}'][n]['tpc_vf_dist'] = list(tpc_vf_dist)
            # datafin[f'data_gen_{mode}'][n]['tpc_vf'] = tpc_vf