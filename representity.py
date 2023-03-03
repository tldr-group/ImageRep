import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np

projects = os.listdir('E:\Dataset')
projects = [f'E:/Dataset/{p}/{p}' for p in projects]
netG = util.load_generator(projects[0])
imgs = []
ls = torch.arange(300, 800)
dicti = {}
for p in projects:
    img = util.generate_image(netG, p)
    if img.any():
        n = p.split('/')[-1]
        dicti[n] = {}
        for i in [0,1]:
            img = 1- img
            vf = torch.mean(img).cpu()
            err_exp = util.real_image_stats(img, ls, vf)
            # print('done exp')
            err_model, ls_model, fac = util.fit_fac(err_exp, ls, vf)
            # print('done fit')
            plt.figure()
            plt.plot(ls, err_exp)
            plt.plot(ls_model, err_model)
            plt.savefig(f'plots/{n}_{i}.png')
            plt.close()
            dicti[n][i] = {'fac':fac,
                        'err_exp': [e.item() for e in err_exp],
                        'err_model': [e.item() for e in err_model],
                        'ls': list(np.array(ls, dtype=np.float64)),
                        'ls_model': list(ls_model),
                        'vf':vf.item(),
                        }
        print(f'{n} done')
with open("data", "w") as fp:
    json.dump(dicti,fp) 