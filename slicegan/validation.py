import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np

# Dataset path and list of subfolders
projects = os.listdir('D:\Dataset')
projects = [f'D:/Dataset/{p}/{p}' for p in projects]
# Load generator
netG = util.load_generator(projects[0])
ls = torch.arange(300, 800, 5)
imgs = []

with open("data.json", "r") as fp:
    datafin = json.load(fp)
data_val = {}
params = datafin['fitted_params']
for j, p in enumerate(projects):
    print(f'{j}/{len(projects)} done')
    img = util.generate_image(netG, p)
    if img.any():
        l = 1000
        n = p.split('/')[-1]
        img = 1- img
        vf = torch.mean(img).cpu()
        sa_img = util.make_sa(img)
        sa = torch.mean(sa_img).cpu()
        # Do calcs on single image
        pred_err_vf, _  = util.make_error_prediction(img[0, :l, :l].cpu(), params['vf'])
        pred_err_sa, _ = util.make_error_prediction(sa_img[0, :l, :l].cpu(), params['sa'])

        err_exp_vf = util.real_image_stats(img, [l], vf)[0]
        err_exp_sa = util.real_image_stats(sa_img, [l], sa)[0]
        
        print(pred_err_vf, err_exp_vf)
        data_val[n] = {'pred_err_vf': pred_err_vf.astype(np.float64),
                   'pred_err_sa':pred_err_sa.astype(np.float64),
                   'err_exp_vf':err_exp_vf.item(),
                   'err_exp_sa':err_exp_sa.item()
                    }
        datafin['validation_data'] = data_val
        with open("data2.json", "w") as fp:
            json.dump(datafin, fp) 