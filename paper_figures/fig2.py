import os
import matplotlib.pyplot as plt
import tifffile
import json
import numpy as np

# import kneed
from scipy.optimize import curve_fit
from scipy import stats
import torch
from representativity.old import util
from torch.nn.functional import interpolate

with open("data_gen2.json", "r") as fp:
    data = json.load(fp)["generated_data"]

l = len(list(data.keys()))
# l=3
c = [(0, 0, 0), (0.5, 0.5, 0.5)]
# plotting = [f'microstructure{f}' for f in [235, 209,205,177]]
plotting = [f"microstructure{f}" for f in [235, 228, 205, 177]]

# plotting = [k for k in data.keys()]
l = len(plotting)
fig, axs = plt.subplots(l, 3)
fig.set_size_inches(12, l * 4)
preds = [[], []]
irs = [[], []]
sas = []
i = 0
for n in list(data.keys()):
    if n not in plotting:
        continue
    img = tifffile.imread(f"/home/amir/microlibDataset/{n}/{n}.tif")
    d = data[n]
    d1 = data[n]

    csets = [["black", "black"], ["gray", "gray"]]
    for j, met in enumerate(["vf", "sa"]):
        cs = csets[j]
        img_dims = [np.array([int(im_len)] * 2) for im_len in d["ls"]]
        ns = util.ns_from_dims(img_dims, d[f"ir_{met}"])
        berns_vf = util.bernouli(d[f"{met}"], ns)
        axs[i, 1].scatter(
            d["ls"],
            d[f"err_exp_{met}"],
            c=cs[0],
            s=8,
            marker="x",
            label=f"{met} errors from sampling",
        )
        axs[i, 1].plot(d["ls"], berns_vf, c=cs[0], label=f"{met} errors from fitted IR")
        # axs[i, 1].plot(d[f'ls'], d[f'err_model_{met}'], c=cs[0], label = f'{met} errors from bernouli')
        y = d[f"tpc_{met}"][
            0
        ]  # TODO write that in sa tpc, only the first direction is shown, or do something else, maybe normalise the tpc? We can do sum, because that's how we calculate the ir!
        x = d[f"tpc_{met}_dist"]
        y = np.array(y)

        # TODO erase this afterwards:
        if met == "vf":
            ir = np.round(d[f"ir_vf"], 1)
            axs[i, 2].plot(x, y, c=cs[1], label=f"Volume fraction 2PC")
            axs[i, 2].axhline(d["vf"] ** 2, linestyle="dashed", label="$p^2$")
            axs[i, 2].plot(
                [0, ir],
                [d["vf"] ** 2 - 0.02, d["vf"] ** 2 - 0.02],
                c="green",
                linewidth=3,
                label=r"$\tilde{a}_2$",
            )
            ticks = [0, int(ir), 20, 40, 60, 80, 100]
            ticklabels = map(str, ticks)
            axs[i, 2].set_xticks(ticks)
            axs[i, 2].set_xticklabels(ticklabels)
            axs[i, 2].fill_between(
                x, d["vf"] ** 2, y, alpha=0.5, label=f"Integrated part"
            )
            axs[i, 2].legend()

        # axs[i, 2].scatter(x[knee], y_data[knee]/y.max(), c =cs[1], marker = 'x', label=f'{met} ir from tpc', s=100)
        ir = d[f"ir_{met}"]
        pred_ir = util.tpc_to_ir(d[f"tpc_{met}_dist"], d[f"tpc_{met}"])
        pred_ir = pred_ir * 1.61
        # axs[i, 2].scatter(x[round(pred_ir)], y[round(pred_ir)], c =cs[1], marker = 'x', label=f'{met} predicted tpc IR', s=100)
        # axs[i, 2].scatter(x[round(ir)], y[round(ir)], facecolors='none', edgecolors = cs[1], label=f'{met} fitted IR', s=100)

        irs[j].append(ir)
        if i == 0:
            axs[i, 1].legend()
            axs[i, 2].legend()
        axs[i, 1].set_xlabel("Image length size [pixels]")
        axs[i, 1].set_ylabel("Volume fraction percentage error [%]")
        axs[i, 2].set_ylabel("2-point correlation function")
    ir = np.round(d[f"ir_vf"], 2)
    sas.append(d["sa"])
    im = img[0] * 255
    si_size, nirs = 160, 5
    sicrop = int(ir * nirs)
    print(ir, sicrop)
    subim = torch.tensor(im[-sicrop:, -sicrop:]).unsqueeze(0).unsqueeze(0).float()
    subim = interpolate(subim, size=(si_size, si_size), mode="nearest")[0, 0]
    subim = np.stack([subim] * 3, axis=-1)

    subim[:5, :, :] = 125
    subim[:, :5, :] = 125
    # subim[5:20, 5:50, :]
    subim[10:15, 10 : 10 + si_size // nirs, :] = 0
    subim[10:15, 10 : 10 + si_size // nirs, 1:-1] = 125

    im = np.stack([im] * 3, axis=-1)
    im[-si_size:, -si_size:] = subim
    axs[i, 0].imshow(im)
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([])
    axs[i, 0].set_ylabel(f"M{n[1:]}")
    axs[i, 0].set_xlabel(
        f"Volume fraction "
        + r"$\tilde{a}_2$: "
        + f"{ir}   Inset mag: x{np.round(si_size/sicrop, 2)}"
    )

    i += 1


plt.tight_layout()
plt.savefig("fig2.pdf", format="pdf")

# fig, axs = plt.subplots(1,2)
# fig.set_size_inches(10,5)
# for i in range(2):
#     ax = axs[i]
#     y=np.array(preds[i])
#     targ = np.array(irs[i])
#     ax.plot(np.arange(60), np.arange(60), c='g')
#     ax.scatter(targ, y, s=5, c='b')
#     coefs_poly3d, _ = curve_fit(util.linear_fit, y, targ)
#     y_data = util.linear_fit(np.arange(60),*coefs_poly3d)
#     ax.plot(y_data, np.arange(60), c='r')
#     y_data = util.linear_fit(y,*coefs_poly3d)

#     ax.set_aspect(1)
#     ax.set_xlabel('ir from model')
#     label = 'True ir' if i==0 else 'SA ir'
#     ax.set_ylabel(label)
#     ax.set_xlim(0,60)
#     ax.set_ylim(0,60)
#     res = np.mean(abs(y-targ)/targ)*100
#     err = abs(y_data-targ)/targ
#     res2 = np.mean(err)*100
#     idx = np.argpartition(err, -3)[-3:]
#     for j in idx:
#         print(list(data.keys())[j])
#     # res = 100*(np.mean((y-targ)**2/targ))**0.5

#     print(res,res2)
