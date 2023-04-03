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
data = {}

for j, p in enumerate(projects):
    print(f'{j}/{len(projects)} done')
    img = util.generate_image(netG, p)
    if img.any():
        n = p.split('/')[-1]
        img = 1- img
        data[n] = {}
        # Fitting
        vf = torch.mean(img).cpu()
        sa_img = util.make_sa(img)
        sa = torch.mean(sa_img).cpu()
        tpc_vf = util.tpc_radial(img)
        tpc_sa = util.tpc_radial(sa_img)
        err_exp_vf = util.real_image_stats(img, ls, vf)
        err_exp_sa = util.real_image_stats(sa_img, ls, sa)
        err_model_vf, ls_model_vf, fac_vf = util.fit_fac(err_exp_vf, ls, vf)
        err_model_sa, ls_model_sa, fac_sa = util.fit_fac(err_exp_sa, ls, sa)
        # Validation
        tpc_vf_val = util.tpc_radial(img, fac_vf, threed=False)
        tpc_sa_val = util.tpc_radial(sa_img, fac_vf, threed=False)
        
        data[n]['fitting'] = {
                    'vf':vf.item(),'sa':sa.item(),
                    'fac_vf':fac_vf,'fac_sa':fac_sa,
                    'tpc_vf':tpc_vf,'tpc_sa':tpc_sa,
                    'err_exp_vf': [e.item() for e in err_exp_vf],
                    'err_exp_sa':[s.item() for s in err_exp_sa],
                    'err_model_vf': [e.item() for e in err_model_vf],
                    'err_model_sa': [e.item() for e in err_model_sa],
                    'ls': list(np.array(ls, dtype=np.float64)),
                    'ls_model_vf': list(ls_model_vf),
                    'ls_model_sa': list(ls_model_sa),
                    }
        data_fin = {'generated_data':data}
        with open("data", "w") as fp:
            json.dump(data_fin, fp) 