import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np

# Dataset path and list of subfolders
projects = os.listdir("D:\Dataset")
projects = [f"D:/Dataset/{p}/{p}" for p in projects]
# Load generator
netG = util.load_generator(projects[0])
edge_lengths = torch.arange(300, 800, 5)
img_dims = [np.array((l, l)) for l in edge_lengths]
imgs = []
data = {}

for j, proj in enumerate(projects):
    img = util.generate_image(netG, proj)
    if img.any():
        microstructure_name = proj.split("/")[-1]
        vf = torch.mean(img).cpu()
        sa_img = util.make_sa(img)
        sa = torch.mean(sa_img).cpu()
        tpc_vf = util.tpc_radial(img)
        tpc_sa = util.tpc_radial(sa_img)
        err_exp_vf = util.real_image_stats(img, edge_lengths, vf)
        err_exp_sa = util.real_image_stats(sa_img, edge_lengths, sa)
        err_model_vf, fac_vf = util.fit_fac(err_exp_vf, img_dims, vf)
        err_model_sa, fac_sa = util.fit_fac(err_exp_sa, img_dims, sa)
        data[microstructure_name] = {
            "vf": vf.item(),
            "sa": sa.item(),
            "fac_vf": fac_vf,
            "fac_sa": fac_sa,
            "tpc_vf": list(tpc_vf),
            "tpc_sa": list(tpc_sa),
            "err_exp_vf": [e.item() for e in err_exp_vf],
            "err_exp_sa": [s.item() for s in err_exp_sa],
            "err_model_vf": [e.item() for e in err_model_vf],
            "err_model_sa": [e.item() for e in err_model_sa],
            "ls": list(np.array(edge_lengths, dtype=np.float64)),
        }
        data_fin = {"generated_data": data}
        print(f"{j}/{len(projects)} done")

        with open("data_gen.json", "w") as fp:
            json.dump(data_fin, fp)
