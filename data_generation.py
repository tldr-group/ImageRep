import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import time

# Dataset path and list of subfolders
with open("micro_names.json", "r") as fp:
    micro_names = json.load(fp)
projects = [f'/home/amir/microlibDataset/{p}/{p}' for p in micro_names]
# Load generator network
netG = util.load_generator(projects[0])
# Edge lengths to test
edge_lengths = torch.arange(300, 800, 5)
# Corresponding dims
img_dims = [np.array((l, l)) for l in edge_lengths]
imgs = []
data = {}

num_projects = 3
projects = projects[:num_projects]

time0 = time.time()
for j, proj in enumerate(projects):
    # Make an image of micro
    img = util.generate_image(netG, proj)
    print(img.size())
    if img.any():
        microstructure_name = proj.split("/")[-1]
        vf = torch.mean(img).cpu()
        sa_img = util.make_sa(img)
        sa = torch.mean(sa_img).cpu()
        tpc_vf_dist, tpc_vf = util.tpc_radial(img)
        tpc_sa_dist, tpc_sa = util.tpc_radial(sa_img)
        err_exp_vf = util.real_image_stats(img, edge_lengths, vf)
        err_exp_sa = util.real_image_stats(sa_img, edge_lengths, sa)
        err_model_vf, fac_vf = util.fit_fac(err_exp_vf, img_dims, vf)
        err_model_sa, fac_sa = util.fit_fac(err_exp_sa, img_dims, sa)
        data[microstructure_name] = {
            "vf": vf.item(),
            "sa": sa.item(),
            "fac_vf": fac_vf,
            "fac_sa": fac_sa,
            "tpc_vf_dist": list(tpc_vf_dist),
            "tpc_vf": list(tpc_vf),
            "tpc_sa_dist": list(tpc_sa_dist),
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
time1 = time.time()

total_time = time1-time0
print(total_time)
project_time = total_time*78/num_projects
print(project_time)