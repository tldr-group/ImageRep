import os
import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import time

mode = '2D'
# Dataset path and list of subfolders
# with open("micro_names.json", "r") as fp:  # TODO change this later
    # micro_names = json.load(fp)
plotting = [f'microstructure{f}' for f in [228, 235,205,177]]
projects = [f'/home/amir/microlibDataset/{p}/{p}' for p in plotting]
# Load generator network
netG = util.load_generator(projects[0])
# Edge lengths to test
edge_lengths = torch.arange(300, 800, 5)
# Corresponding dims
img_dims = [np.array((l, l)) for l in edge_lengths]
imgs = []
data = {}

num_projects = len(projects)
projects = projects[:num_projects]

l = 1000 if mode=='2D' else 400
time0 = time.time()
for j, proj in enumerate(projects):
    # Make an image of micro
    img = util.generate_image(netG, proj)
    print(img.size())
    if img.any():
        microstructure_name = proj.split("/")[-1]
        vf = torch.mean(img).cpu().item()
        sa_images = util.make_sas(img)
        img = [img]
        sa = torch.mean(util.sa_map_from_sas(sa_images)).cpu().item()
        print(f'starting tpc')
        testimg = [img[0, :l, :l].cpu() if mode=='2D' else img[0, :l, :l, :l].cpu() for img in img]
        tpc_vf_dist, tpc_vf = util.tpc_radial(testimg)
        sa_testimg = [sa_img[0, :l, :l].cpu() if mode=='2D' else sa_img[0, :l, :l, :l].cpu() for sa_img in sa_images]
        tpc_sa_dist, tpc_sa = util.tpc_radial(sa_testimg)
        print(f'finished tpc, starting image stats')
        err_exp_vf = util.real_image_stats(img[0], edge_lengths, vf, repeats=2000)
        err_exp_sa = util.real_image_stats(util.sa_map_from_sas(sa_images), edge_lengths, sa, repeats=2000)
        ir_vf = util.fit_ir(err_exp_vf, img_dims, vf)
        shape_vf = [np.array(testimg[0].size())]
        err_model_vf = util.bernouli(vf, util.ns_from_dims(shape_vf, ir_vf), conf=0.95)
        ir_sa = util.fit_ir(err_exp_sa, img_dims, sa)
        shape_sa = [np.array(sa_testimg[0].size())]
        err_model_sa = util.bernouli(sa, util.ns_from_dims(shape_sa, ir_sa), conf=0.95)
        data[microstructure_name] = {
            "vf": vf,
            "sa": sa,
            "ir_vf": ir_vf,
            "ir_sa": ir_sa,
            "tpc_vf_dist": list(tpc_vf_dist),  
            "tpc_vf": [list(tpc) for tpc in tpc_vf],
            "tpc_sa_dist": list(tpc_sa_dist),
            "tpc_sa": [list(tpc) for tpc in tpc_sa],
            "err_exp_vf": [e.item() for e in err_exp_vf],
            "err_exp_sa": [s.item() for s in err_exp_sa],
            "err_model_vf": [e.item() for e in err_model_vf],
            "err_model_sa": [e.item() for e in err_model_sa],
            "ls": list(np.array(edge_lengths, dtype=np.float64)),
        }
        data_fin = {"generated_data": data}
        print(f"{j}/{len(projects)} done")

        with open("data_gen2.json", "w") as fp:
            json.dump(data_fin, fp)
time1 = time.time()

total_time = time1-time0
print(total_time)
project_time = total_time*78/num_projects
print(project_time)