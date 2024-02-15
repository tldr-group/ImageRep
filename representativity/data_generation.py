import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import time

'''
File: data_generation.py

Description: This script is used to generate the statistics for the representativity of 
microstructures from the microlib dataset. 
The statistics generated are then saved in a json file, to be analyzed and used for 
representativity prediction of a new microstructure or micrograph.
'''


with open("microlib_statistics.json", "r") as fp:
    datafin = json.load(fp)
# Dataset path and list of subfolders
with open("micro_names.json", "r") as fp:
    micro_names = json.load(fp)
projects = [f'/home/amir/microlibDataset/{p}/{p}' for p in micro_names]
# Load generator
netG = util.load_generator(projects[0])

modes = ['3D']

# Edge lengths for the experimental statistical analysis:
datafin['validation_data2D']['edge_lengths_exp'] = torch.arange(500, 1000, 20)
datafin['validation_data3D']['edge_lengths_exp'] = torch.arange(120, 400, 20)

# Edge lengths for the predicted integral range:
datafin['validation_data2D']['edge_lengths_pred'] = torch.arange(8, 65, 4)*32
datafin['validation_data3D']['edge_lengths_pred'] = torch.arange(4, 19, 1)*32

# data_val = {}
for j, p in enumerate(projects):
    for mode in modes:
        edge_lengths_exp = datafin[f'validation_data{mode}']['edge_lengths_exp']
        edge_lengths_pred = datafin[f'validation_data{mode}']['edge_lengths_pred']
        t_before = time.time()
        pred_err_vfs, pred_ir_vfs = [], []
        # exp_err_vfs, exp_ir_vfs = [], []
        # pred_err_sas, pred_ir_sas = [], []
        # exp_err_sas, exp_ir_sas = [], []
        # dims_i = [0] if mode == '3D' else [0,1,2]  # for 2D and 3D comparison
        dims_i = [0] 
        for dim_i in dims_i:
            print(f'{j}/{len(projects)} done')
            
            lf = 21 if mode=='3D' else 70
            img = util.generate_image(netG, p, dim_i, lf=lf, threed=mode=='3D', reps=1)
            print(mode)
            print(img.size())
            if img.any():
                # testing single image of edge length l
                
                l = img.size()[-1] 
                print(img.size())
                print(l)
                n = p.split('/')[-1]
                # make the sa images.
                print(n)
                vf = torch.mean(img).cpu().item()
                img = [img]  # this way it will work together with the surface areas.
                
                print(f'{j} vf = {vf}')
                testimg = [img[0, :l, :l].cpu() if mode=='2D' else img[0, :l, :l, :l].cpu() for img in img]
                pred_err_vf, _, tpc_vf_dist, tpc_vf, pred_ir_vf = util.make_error_prediction(testimg, model_error=False, correction=False, mxtpc=100)
                compared_shape = [np.array(testimg[0].size())]
                exp_err_vf, exp_ir_vf = util.stat_analysis_error(img[0], edge_lengths_exp, compared_shape, conf=0.95)
                pred_err_vfs.append(pred_err_vf)
                pred_ir_vfs.append(pred_ir_vf)
                # exp_err_vfs.append(exp_err_vf)
                # exp_ir_vfs.append(exp_ir_vf)
                # print(f'{j} pred error vf = {pred_err_vf*100}')
                # print(f'{j} experiment error vf = {err_exp_vf}')
                # print(f'% diff vf = {(err_exp_vf-pred_err_vf*100)/err_exp_vf}')
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
                # err_exp_sa, exp_ir_sa = util.stat_analysis_error(util.sa_map_from_sas(sa_images), edge_lengths, img_dims, compared_shape, conf=0.95)
                # print(f'{j} pred error = {pred_err_sa*100}')
                # print(f'{j} experiment error = {err_exp_sa}')
                # print(f'% diff = {(err_exp_sa-pred_err_sa*100)/err_exp_sa}')

                # pred_err_sas.append(pred_err_sa)
                # pred_ir_sas.append(pred_ir_sa)
                # exp_err_sas.append(err_exp_sa)
                # exp_ir_sas.append(exp_ir_sa)

                # print(f'{j} starting real image stats')
                # Do stats on the full generated image
                
                # err_exp_vf = util.real_image_stats(img[0], edge_lengths, vf)[0]  
                # err_exp_sa = util.real_image_stats(util.sa_map_from_sas(sa_images), edge_lengths, sa)[0]  # TODO calc with new sa
                
                # tpc_vf = [list(tpc) for tpc in tpc_vf]
                # tpc_sa = [list(tpc) for tpc in tpc_sa]
                # data_val[n] = {'pred_err_vf': pred_err_vf.astype(np.float64)*100,
                        #    'pred_err_sa':pred_err_sa.astype(np.float64)*100,
                        #    'err_exp_vf':err_exp_vf.item(),
                        #    'err_exp_sa':err_exp_sa.item(),
                            # 'tpc_vf_dist':list(tpc_vf_dist),
                            # 'tpc_vf':tpc_vf,
                            # 'tpc_sa_dist':list(tpc_sa_dist),
                            # 'tpc_sa':tpc_sa
                            # }
                
        if pred_ir_vfs:
            # print(errs_ir_vf)
            # exp_ir_vf = np.mean(errs_ir_vf)
        #     print(f'mean ir = {exp_ir_vf}')
            # datafin[f'validation_data{mode}'] = data_val
            # datafin[f'validation_data{mode}'][n]['sa'] = sa
            # datafin[f'validation_data{mode}'][n]['pred_err_sa'] = pred_err_sa.astype(np.float64)*100
            # datafin[f'validation_data{mode}'][n]['pred_ir_sa'] = pred_ir_sa
            # datafin[f'validation_data{mode}'][n]['err_exp_sa'] = err_exp_sa.item()
            # datafin[f'validation_data{mode}'][n]['exp_ir_sa'] = exp_ir_sa

            # datafin[f'validation_data{mode}'][n]['tpc_sa_dist'] = list(tpc_sa_dist)
            # datafin[f'validation_data{mode}'][n]['tpc_sa'] = tpc_sa
            
            datafin[f'validation_data{mode}'][n]['vf'] = vf
            print(pred_ir_vfs)
            print(f'mean ir = {np.mean(pred_ir_vfs)}')
            datafin[f'validation_data{mode}'][n]['pred_ir_vf'] = np.mean(pred_ir_vfs)
            datafin[f'validation_data{mode}'][n]['pred_ir_vf'] = np.mean(pred_ir_vfs)
            if mode=='3D':
                datafin[f'validation_data{mode}'][n]['dim_variation'] = 0
            else:
                datafin[f'validation_data{mode}'][n]['dim_variation'] = np.var(pred_ir_vfs)/np.mean(pred_ir_vfs)
                print(f'dim variation = {np.var(pred_ir_vfs)/np.mean(pred_ir_vfs)}')
            datafin[f'validation_data{mode}'][n]['pred_err_vf'] = np.mean(pred_err_vfs).astype(np.float64)*100

            # print(pred_ir_sas)
            # print(f'mean ir = {np.mean(pred_ir_sas)}')
            # datafin[f'validation_data{mode}'][n]['pred_ir_sa'] = np.mean(pred_ir_sas)
            # if mode=='3D':
            #     datafin[f'validation_data{mode}'][n]['dim_variation'] = 0
            # else:
            #     datafin[f'validation_data{mode}'][n]['dim_variation'] = np.var(pred_ir_sas)/np.mean(pred_ir_sas)
            #     print(f'dim variation = {np.var(pred_ir_sas)/np.mean(pred_ir_sas)}')
            # datafin[f'validation_data{mode}'][n]['pred_err_sa'] = np.mean(pred_err_sas).astype(np.float64)*100

            # print(exp_ir_vfs)
            # print(f'mean ir = {np.mean(exp_ir_vfs)}')
            # datafin[f'validation_data{mode}'][n]['exp_ir_vf'] = np.mean(exp_ir_vfs)
            # datafin[f'validation_data{mode}'][n]['exp_ir_vf'] = np.mean(exp_ir_vfs)
            # if mode=='3D':
            #     datafin[f'validation_data{mode}'][n]['exp_dim_variation'] = 0
            # else:
            #     datafin[f'validation_data{mode}'][n]['exp_dim_variation'] = np.var(exp_ir_vfs)/np.mean(exp_ir_vfs)
            #     print(f'dim variation = {np.var(exp_ir_vfs)/np.mean(exp_ir_vfs)}')
            # datafin[f'validation_data{mode}'][n]['exp_err_vf'] = np.mean(exp_err_vfs).astype(np.float64)

            # print(exp_ir_sas)
            # print(f'mean ir = {np.mean(exp_ir_sas)}')
            # datafin[f'validation_data{mode}'][n]['exp_ir_sa'] = np.mean(exp_ir_sas)
            # if mode=='3D':
            #     datafin[f'validation_data{mode}'][n]['exp_dim_variation'] = 0
            # else:
            #     datafin[f'validation_data{mode}'][n]['exp_dim_variation'] = np.var(exp_ir_sas)/np.mean(exp_ir_sas)
            #     print(f'dim variation = {np.var(exp_ir_sas)/np.mean(exp_ir_sas)}')
            # datafin[f'validation_data{mode}'][n]['exp_err_sa'] = np.mean(exp_err_sas).astype(np.float64)

            # datafin[f'validation_data{mode}'][n]['err_exp_vf'] = err_exp_vf.item()
            # datafin[f'validation_data{mode}'][n]['exp_ir_vf'] = exp_ir_vf
            # datafin[f'validation_data{mode}'][n]['tpc_vf_dist'] = list(tpc_vf_dist)
            # datafin[f'validation_data{mode}'][n]['tpc_vf'] = tpc_vf
            
            print()
            with open(f"microlib_rep_data.json", "w") as fp:
                json.dump(datafin, fp) 
            print(f'time for one micro {mode} = {np.round((time.time()-t_before)/60)} minutes')

if __name__ == '__main__':
    pass