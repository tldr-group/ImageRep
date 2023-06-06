import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import time


mode = '3D'
# Dataset path and list of subfolders
with open("micro_names.json", "r") as fp:
    micro_names = json.load(fp)
projects = [f'/home/amir/microlibDataset/{p}/{p}' for p in micro_names]
# Load generator
netG = util.load_generator(projects[0])
imgs = []

# Edge lengths to test
edge_lengths = [torch.arange(300, 800, 5), torch.arange(120, 405, 5)]
# Corresponding dims
img_dims_2d = [np.array((l, l)) for l in edge_lengths[0]]
img_dims_3d = [np.array((l, l, l)) for l in edge_lengths[1]]
edge_lengths = edge_lengths[1] if mode=='3D' else edge_lengths[0]
img_dims = img_dims_3d if mode == '3D' else img_dims_2d

num_projects = len(projects)
projects = projects[:num_projects]

with open("datafin_new.json", "r") as fp:
    datafin = json.load(fp)

# data_val = {}
for j, p in enumerate(projects):
    t_before = time.time()
    print(f'{j}/{len(projects)} done')
    lf = 16 if mode=='3D' else 50
    img = util.generate_image(netG, p, lf=lf, threed=mode=='3D')
    print(img.size())
    if img.any():
        # testing single image of edge length l
        l = 1000 if mode=='2D' else 400
        n = p.split('/')[-1]
        # make the sa images
        print(f'{j} starting to sa')
        sa_images = util.make_sas(img)
        # Do calcs on single image
        print(f'{j} starting to testing error')
        sa = torch.mean(util.sa_map_from_sas(sa_images)).cpu().item()
        # vf = torch.mean(img).cpu().item()
        # img = [img]  # this way it will work together with the surface areas
        # print(f'{j} vf = {vf}')
        print(f'{j} sa = {sa}')
        # testimg = [img[0, :l, :l].cpu() if mode=='2D' else img[0, :l, :l, :l].cpu() for img in img]
        sa_testimg = [sa_img[0, :l, :l].cpu() if mode=='2D' else sa_img[0, :l, :l, :l].cpu() for sa_img in sa_images]
        # pred_err_vf, _, tpc_vf_dist, tpc_vf = util.make_error_prediction(testimg, vf, model_error=False, correction=False)
        pred_err_sa, _, tpc_sa_dist, tpc_sa = util.make_error_prediction(sa_testimg, sa, model_error=False, correction=False)
        print(f'{j} pred error = {pred_err_sa*100}')
        print(f'{j} starting real image stats')
        # Do stats on the full generated image
        # err_exp_vf = util.stat_analysis_error(img[0], edge_lengths, img_dims, vf, threed=mode=='3D', conf=0.95)
        err_exp_sa = util.stat_analysis_error(util.sa_map_from_sas(sa_images), edge_lengths, img_dims, sa, threed=mode=='3D', conf=0.95)
        
        # err_exp_vf = util.real_image_stats(img[0], edge_lengths, vf, threed=mode=='3D')[0]  
        # err_exp_sa = util.real_image_stats(util.sa_map_from_sas(sa_images), edge_lengths, sa, threed=mode=='3D')[0]  # TODO calc with new sa
        print(f'{j} experiment error = {err_exp_sa}')
        print(f'% diff = {(err_exp_sa-pred_err_sa*100)/err_exp_sa}')
        # tpc_vf = [list(tpc) for tpc in tpc_vf]
        tpc_sa = [list(tpc) for tpc in tpc_sa]
        # data_val[n] = {'pred_err_vf': pred_err_vf.astype(np.float64)*100,
                #    'pred_err_sa':pred_err_sa.astype(np.float64)*100,
                #    'err_exp_vf':err_exp_vf.item(),
                #    'err_exp_sa':err_exp_sa.item(),
                    # 'tpc_vf_dist':list(tpc_vf_dist),
                    # 'tpc_vf':tpc_vf,
                    # 'tpc_sa_dist':list(tpc_sa_dist),
                    # 'tpc_sa':tpc_sa
                    # }
        # datafin[f'validation_data{mode}'] = data_val
        # datafin[f'validation_data{mode}'][n]['err_exp_vf'] = err_exp_vf
        datafin[f'validation_data{mode}'][n]['pred_err_sa'] = pred_err_sa.astype(np.float64)*100
        datafin[f'validation_data{mode}'][n]['err_exp_sa'] = err_exp_sa.item()
        datafin[f'validation_data{mode}'][n]['tpc_sa_dist'] = list(tpc_sa_dist)
        datafin[f'validation_data{mode}'][n]['tpc_sa'] = tpc_sa
        # datafin[f'validation_data{mode}'][n]['vf'] = vf
        datafin[f'validation_data{mode}'][n]['sa'] = sa
        with open(f"datafin_new2.json", "w") as fp:
            json.dump(datafin, fp) 
    print(f'time for one micro {mode} = {np.round((time.time()-t_before)/60)} minutes')

    