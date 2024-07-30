import os
import matplotlib.pyplot as plt
import tifffile
import json
import numpy as np

# import kneed
from scipy.optimize import curve_fit
from scipy import stats
import torch
from representativity import core, util
from representativity.correction_fitting import microlib_statistics
from mpl_toolkits.axes_grid1 import make_axes_locatable

# with open("data_gen2.json", "r") as fp:
#     data = json.load(fp)["generated_data"]

# l=len(list(data.keys()))
# l=3
c = [(0, 0, 0), (0.5, 0.5, 0.5)]
# plotting = [f'microstructure{f}' for f in [235, 209,205,177]]
plotting_ims = [f"microstructure{f}" for f in [235, 228, 205, 177]]

# plotting = [k for k in data.keys()]
l = len(plotting_ims)
fig, axs = plt.subplots(l, 3)
fig.set_size_inches(12, l * 3.5)
preds = [[], []]
irs = [[], []]
colors = {"cls": "tab:orange", "stds": "tab:green"}

all_data, micros, netG, v_names, run_v_names = microlib_statistics.json_preprocessing()
lens_for_fit = list(
    range(500, 1000, 20)
)  # lengths of images for fitting L_characteristic
plotting_ims = [micro for micro in micros if micro.split("/")[-1] in plotting_ims]
# run the statistical analysis on the microlib dataset
for i, p in enumerate(plotting_ims):

    try:
        netG.load_state_dict(torch.load(p + "_Gen.pt"))
    except:  # if the image is greayscale it's excepting because there's only 1 channel
        continue
    imsize = 1600
    lf = imsize // 32 + 2  # the size of G's input
    many_images = util.generate_image(netG, lf=lf, threed=False, reps=10)
    pf = many_images.mean().cpu().numpy()
    small_imsize = 512
    img = many_images[0].detach().cpu().numpy()[:small_imsize, :small_imsize]

    csets = [["black", "black"], ["gray", "gray"]]
    conf = 0.95
    errs = util.real_image_stats(many_images, lens_for_fit, pf, conf=conf)
    sizes_for_fit = [[lf, lf] for lf in lens_for_fit]
    real_cls = core.fit_statisical_cls_from_errors(errs, sizes_for_fit, pf)  # type: ignore
    stds = errs / stats.norm.interval(conf)[1] * pf / 100
    std_fit = (real_cls**2 / (np.array(lens_for_fit) ** 2) * pf * (1 - pf)) ** 0.5
    # print(stds)
    vars = stds**2
    # from variations to L_characteristic using image size and phase fraction
    clss = (np.array(lens_for_fit) ** 2 * vars / pf / (1 - pf)) ** 0.5
    print(clss)

    axs_twin = axs[i, 1].twinx()
    stds_scatter = axs_twin.scatter(
        lens_for_fit, stds, s=8, color=colors["stds"], label=f"Standard deviations"
    )
    twin_fit = axs_twin.plot(
        lens_for_fit, std_fit, color=colors["stds"], label=f"Standard deviations fit"
    )
    axs_twin.tick_params(axis="y", labelcolor=colors["stds"])
    axs_twin.set_ylim(0, 0.025)
    cls_scatter = axs[i, 1].scatter(
        lens_for_fit,
        clss,
        s=8,
        color=colors["cls"],
        label=f"Characteristic length scales",
    )
    cls_fit = axs[i, 1].hlines(
        real_cls,
        lens_for_fit[0],
        lens_for_fit[-1],
        color=colors["cls"],
        label=f"Characteristic length scales fit",
    )
    axs[i, 1].tick_params(axis="y", labelcolor=colors["cls"])
    axs[i, 1].set_ylim(0, 28)

    dims = len(img.shape)
    # print(f'starting tpc radial')
    tpc = core.radial_tpc(img, volumetric=False)
    cls = core.tpc_to_cls(tpc, img)

    contour = axs[i, 2].contourf(tpc, cmap="plasma", levels=200)
    divider = make_axes_locatable(axs[i, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(contour, cax=cax, orientation="vertical")

    if i == 0:
        plots = [stds_scatter, twin_fit[0], cls_scatter, cls_fit]
        label_plots = [plot.get_label() for plot in plots]
        axs[i, 1].legend(plots, label_plots)

    axs[i,2].legend()
    # axs[i,1].set_xlabel('Image length size [pixels]')
    # axs[i,1].set_ylabel('Volume fraction percentage error [%]')
    # axs[i,2].set_ylabel('2-point correlation function')
    # ir = np.round(d[f'ir_vf'], 2)
    # im = img[0]*255
    # si_size, nirs = 160, 5
    # sicrop = int(ir*nirs)
    # print(ir, sicrop)
    # subim=torch.tensor(im[-sicrop:,-sicrop:]).unsqueeze(0).unsqueeze(0).float()
    # subim = interpolate(subim, size=(si_size,si_size), mode='nearest')[0,0]
    # subim = np.stack([subim]*3, axis=-1)

    # subim[:5,:,:] = 125
    # subim[:,:5,:] = 125
    # # subim[5:20, 5:50, :]
    # subim[10:15, 10:10+si_size//nirs, :] = 0
    # subim[10:15, 10:10+si_size//nirs, 1:-1] = 125

    # im = np.stack([im]*3, axis=-1)
    # im[-si_size:,-si_size:] = subim
    # axs[i, 0].imshow(im)
    # axs[i, 0].set_xticks([])
    # axs[i, 0].set_yticks([])
    # axs[i, 0].set_ylabel(f'M{n[1:]}')
    # axs[i, 0].set_xlabel(f'Volume fraction '+ r'$\tilde{a}_2$: '+ f'{ir}   Inset mag: x{np.round(si_size/sicrop, 2)}')

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
