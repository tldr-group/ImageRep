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
from scipy.ndimage import zoom 

# with open("data_gen2.json", "r") as fp:
#     data = json.load(fp)["generated_data"]

# l=len(list(data.keys()))
# l=3
c = [(0, 0, 0), (0.5, 0.5, 0.5)]
# plotting = [f'microstructure{f}' for f in [235, 209,205,177]]
plotting_nums = [235, 228, 205, 177]
plotting_ims = [f"microstructure{f}" for f in plotting_nums]

# plotting = [k for k in data.keys()]
l = len(plotting_ims)
fig, axs = plt.subplots(l, 3)
fig.set_size_inches(12, l * 3.5)
preds = [[], []]
irs = [[], []]
colors = {"pred": "tab:orange", "true": "tab:green"}

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
    many_images = util.generate_image(netG, lf=lf, threed=False, reps=150)
    many_images = many_images.detach().cpu().numpy()
    pf = many_images.mean()
    small_imsize = 512
    img = many_images[0][:small_imsize, :small_imsize]

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

    # axs_twin = axs[i, 1].twinx()
    axs[i, 1].scatter(
        lens_for_fit, stds, color='black', s=8,  label=f"Standard deviations"
    )
    axs[i, 1].plot(
        lens_for_fit, std_fit, color=colors["true"], label=f"Best fit using the characteristic\nlength scale: {np.round(real_cls, 2)}"
    )
    # axs[i, 1].tick_params(axis="y", labelcolor=colors["stds"])
    axs[i, 1].set_ylim(0, 0.025)
    axs[i, 1].legend(loc='upper right')

    center = 40
    dims = len(img.shape)
    # print(f'starting tpc radial')
    tpc = core.radial_tpc(img, volumetric=False)
    center_im = small_imsize // 2
    tpc_im = tpc[center_im-center:center_im+center, center_im-center:center_im+center]
    img_pf = img.mean()
    cls = core.tpc_to_cls(tpc, img)

    contour = axs[i, 2].contourf(tpc_im, cmap="plasma", levels=200)
    for c in contour.collections:
        c.set_edgecolor("face")
    circle_real = plt.Circle((center, center), real_cls, fill=False, color=colors["true"], label=f"True Char. l. s. radius: {np.round(real_cls, 2)}")
    circle_pred = plt.Circle((center, center), cls, fill=False, color=colors["pred"], label=f"Predicted Char. l. s. radius: {np.round(cls, 2)}")
    axs[i, 2].add_artist(circle_real)
    axs[i, 2].add_artist(circle_pred)
    x_ticks = axs[i, 2].get_xticks()[1:-1]
    axs[i, 2].set_xticks(x_ticks, np.int64(np.array(x_ticks) - center))
    axs[i, 2].set_yticks(x_ticks, np.int64(np.array(x_ticks) - center))
    divider = make_axes_locatable(axs[i, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contour, cax=cax, orientation="vertical")
    # cbar_ticks = cbar.ax.get_yticks()
    cbar_ticks = np.linspace(img_pf, img_pf**2, 6)
    cbar.ax.set_yticks(cbar_ticks, [r'$\Phi(\omega)$']+list(np.round(cbar_ticks[1:-1],2))+[r'$\Phi(\omega)^2$'])
    cbar.set_label(f'Two-point correlation function')
    fakexy = [0, 0]
    circle_pred = plt.Line2D(fakexy, fakexy, linestyle='none', marker='o', fillstyle='none', color=colors["pred"], alpha=1.00)
    circle_real = plt.Line2D(fakexy, fakexy, linestyle='none', marker='o', fillstyle='none', color=colors["true"], alpha=1.00)
    axs[i, 2].legend([circle_real, circle_pred], [f"True Char. l. s.: {np.round(real_cls, 2)}", f"Predicted Char. l. s. from \nimage on the left column: {np.round(cls, 2)}"], loc='upper right')
    axs[i, 2].set_xlabel('Two-point correlation distance')
    axs[i, 2].set_ylabel('Two-point correlation distance')

    axs[i,1].set_xlabel('Image size')
    xticks_middle = axs[i,1].get_xticks()[1:-1]
    axs[i,1].set_xticks(xticks_middle, [f'{int(xtick)}$^2$' for xtick in xticks_middle])
    axs[i,1].set_ylabel(r'Phase fraction standard deviation')
    # axs[i,2].set_ylabel('TPC distance')
    # axs[i,2].set_xlabel('TPC distance')
    # img = img*255
    imshow_size = 512
    img = img[:imshow_size, :imshow_size]
    # 
    si_size, nirs = imshow_size//2, 5
    sicrop = int(real_cls*nirs)
    zoom_mag = si_size/sicrop
    print(real_cls, sicrop)
    subim = img[-sicrop:,-sicrop:]
    subim = zoom(subim, zoom=(zoom_mag,zoom_mag), order=0)
    subim = np.stack([subim]*3, axis=-1)

    boundary_len = 5
    subim[:boundary_len,:,:] = 0.5
    subim[:,:boundary_len,:] = 0.5
    # subim[5:20, 5:50, :]
    subim[10:10+boundary_len, 10:10+si_size//nirs, :] = 0
    subim[10:10+boundary_len, 10:10+si_size//nirs, 1:-1] = 0.5

    img = np.stack([img]*3, axis=-1)
    img[-si_size:,-si_size:] = subim
    # img = np.stack([zoom(img[:,:,i], zoom=4, order=0) for i in range(np.shape(img)[-1])], axis=-1)
    axs[i, 0].imshow(img, cmap="gray", interpolation='nearest')
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([])
    axs[i, 0].set_ylabel(f'Microstructure {plotting_nums[i]}')
    axs[i, 0].set_xlabel(f'    $\Phi(\omega)$: '+ '%.2f' % img_pf + f'          Inset mag: x{np.round(si_size/sicrop, 2)}')



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
