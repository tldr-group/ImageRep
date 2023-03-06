import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np

projects = os.listdir('D:\Dataset')
projects = [f'D:/Dataset/{p}/{p}' for p in projects]
netG = util.load_generator(projects[0])
imgs = []
ls = torch.arange(300, 800)
dicti = {}

for p in projects[2:]:
    img = util.generate_image(netG, p)
    if img.any():
        n = p.split('/')[-1]
        dicti[n] = {}
        for i in [0,1]:
            img = 1- img
            tpc = util.tpc(img)
            vf = torch.mean(img).cpu()
            err_exp = util.real_image_stats(img, ls, vf)
            err_model, ls_model, fac = util.fit_fac(err_exp, ls, vf)
            plt.figure()
            plt.plot(ls, err_exp)
            plt.plot(ls_model, err_model)
            plt.savefig(f'plots/{n}_{i}.png')
            plt.close()
            dicti[n][i] = {
                        'fac':fac,
                        'tpc':tpc,
                        'err_exp': [e.item() for e in err_exp],
                        'err_model': [e.item() for e in err_model],
                        'ls': list(np.array(ls, dtype=np.float64)),
                        'ls_model': list(ls_model),
                        'vf':vf.item(),
                        }
            # dicti[n][str(i)]['tpc'] = tpc
        print(f'{n} done')

with open("data_tpc", "w") as fp:
    json.dump(dicti,fp) 