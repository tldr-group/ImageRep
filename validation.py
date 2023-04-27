import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np


mode = '2D'
# Dataset path and list of subfolders
projects = os.listdir('D:\Dataset')
projects = [f'D:/Dataset/{p}/{p}' for p in projects]
# Load generator
netG = util.load_generator(projects[0])
imgs = []

with open("data.json", "r") as fp:
    datafin = json.load(fp)
data_val = {}
for j, p in enumerate(projects):
    print(f'{j}/{len(projects)} done')
    img = util.generate_image(netG, p)
    if img.any():
        # testing single image of edge length l
        l = 1000 if mode=='2D' else 400
        n = p.split('/')[-1]
        vf = torch.mean(img).cpu()
        # make the sa images
        sa_img = util.make_sa(img)
        sa = torch.mean(sa_img).cpu()
        # Do calcs on single image
        testimg = img[0, :l, :l].cpu() if mode=='2D' else img[0, :l, :l, :l].cpu()
        sa_testimg = img[0, :l, :l].cpu() if mode=='2D' else sa_img[0, :l, :l, :l].cpu()
        pred_err_vf, _, tpc_vf = util.make_error_prediction(testimg, model_error=False, correction=False)
        pred_err_sa, _, tpc_sa = util.make_error_prediction(sa_testimg, model_error=False, correction=False)
        # Do stats on the full generated image
        err_exp_vf = util.real_image_stats(img, [l], vf, threed=mode=='3D')[0]
        err_exp_sa = util.real_image_stats(sa_img, [l], sa, threed=mode=='3D')[0]
        data_val[n] = {'pred_err_vf': pred_err_vf.numpy().astype(np.float64)*100,
                   'pred_err_sa':pred_err_sa.numpy().astype(np.float64)*100,
                   'err_exp_vf':err_exp_vf.item(),
                   'err_exp_sa':err_exp_sa.item(),
                    'tpc_vf':list(tpc_vf),
                    'tpc_sa':list(tpc_sa)
                    }
        datafin[f'validation_data{mode}'] = data_val
        with open(f"data.json", "w") as fp:
            json.dump(datafin, fp) 