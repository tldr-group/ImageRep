import numpy as np

np.random.seed(0)
from tifffile import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, InsetPosition
from types import NoneType

from skimage.color import label2rgb

from representativity import core

plt.rcParams["text.usetex"] = True


img_fig = plt.figure("img", dpi=300)
img_fig.set_size_inches(14, 14)

DEFAULT_MICROSTRUCTURE = imread("tests/resources/default.tiff")[0]
phases = np.unique(DEFAULT_MICROSTRUCTURE)
SELECTED_PHASE = phases[0]
img = np.where(DEFAULT_MICROSTRUCTURE == SELECTED_PHASE, 1, 0)
ih, iw = img.shape

image_phase_fraction = np.mean(img)

tpc = core.radial_tpc(img, volumetric=False)

tpc_l = 40
center_im = ih // 2
tpc_im = tpc[
    center_im - tpc_l : center_im + tpc_l, center_im - tpc_l : center_im + tpc_l
]
cls = core.tpc_to_cls(tpc, img)
middle_idx = np.array(tpc.shape) // 2
raw_dist_arr = np.indices(tpc.shape)
remapped_dist_arr = np.abs((raw_dist_arr.T - middle_idx.T).T)
img_volume = np.prod(img.shape)
norm_vol = (np.array(img.shape).T - remapped_dist_arr.T).T
norm_vol = np.prod(norm_vol, axis=0) / img_volume
euc_dist_arr: np.ndarray = np.sqrt(np.sum(remapped_dist_arr**2, axis=0))
end_idx = core.find_end_dist_tpc(image_phase_fraction, tpc, euc_dist_arr)


middle_slice_tpc = tpc[center_im : center_im + 1, center_im:][0]
x = np.arange(0, len(middle_slice_tpc))
ymax = np.max(middle_slice_tpc)
print(ymax)


TITLE_FS = 48
LABEL_FS = 36
TICK_FS = 30

img_ax = plt.gca()

img_ax.imshow(img, cmap="binary_r")
img_ax.set_xticks([])
img_ax.set_yticks([])
img_ax.set_xlabel(r"$X_{l_1}=" + f"{img.shape[1]}$", fontsize=LABEL_FS)
img_ax.set_ylabel(r"$X_{l_2}=" + f"{img.shape[0]}$", fontsize=LABEL_FS)

img_ax.text(
    200,
    ih + 110,
    r"Phase Fraction $\Phi_{X}(\omega)=" + f"{image_phase_fraction:.4f}" + r"$",
    fontsize=LABEL_FS + 2,
)
img_ax.text(
    200,
    ih + 180,
    r"$X = X_{l_1} \times X_{l_2}, \| X \| =" + f"{img.shape[0] * img.shape[1]}" + r"$",
    fontsize=LABEL_FS + 2,
)

img_ax.set_title(r"Binary Microstructure, $\omega$", fontsize=TITLE_FS)

plt.savefig("paper_figures/workflow/img1.png")
# plt.tight_layout()

tpc_fig = plt.figure("TPC", dpi=300)
tpc_fig.set_size_inches(14, 14)

tpc_ax = plt.gca()

tpc_ax.contourf(tpc_im, cmap="plasma", levels=200)
tpc_ax.set_aspect("equal")
x_ticks = tpc_ax.get_xticks()[1:-1]
tpc_ax.set_xticks(x_ticks, np.int64(np.array(x_ticks) - tpc_l), fontsize=TICK_FS)
tpc_ax.set_yticks(x_ticks, np.int64(np.array(x_ticks) - tpc_l), fontsize=TICK_FS)
tpc_ax.set_xlabel(r"$x$-distance", fontsize=LABEL_FS)
tpc_ax.set_ylabel(r"$y$-distance", fontsize=LABEL_FS)

circle_real = plt.Circle((tpc_l, tpc_l), cls, fill=False, color="red", lw=5, ls="--")
tpc_ax.text(
    tpc_l - 16,
    25 + tpc_l,
    r"CLS, $a_n=" + f"{cls:.2f}" + r"$",
    fontsize=TITLE_FS,
    color="red",
)
tpc_ax.add_artist(circle_real)
tpc_ax.text(
    1.75 * cls,
    0.55 * cls,
    r"Decorrelation at",
    fontsize=TITLE_FS,
    color="black",
)
tpc_ax.text(
    (1.75 + 0.5) * cls,
    0.25 * cls,
    r"$r_o=" + f"{end_idx}" + r"$",
    fontsize=TITLE_FS,
    color="black",
)

tpc_ax.set_title(r"Two-Point Correlation, $T_{r, X}(\omega)$", fontsize=TITLE_FS)

axin_loc = (0.08, 0.08, 0.3, 0.3)
ax0, ay0, aw, ah = axin_loc
axin = tpc_ax.inset_axes(axin_loc)
axin.plot([tpc_l, 1.93 * tpc_l], [tpc_l, tpc_l + 1])
# hide the ticks of the linked axes
axin.set_xticks([])
axin.set_yticks([])
mark_inset(tpc_ax, axin, 2, 1)


axin2 = plt.gcf().add_axes(axin_loc)
print(x.shape, middle_slice_tpc.shape)
axin2.plot(x, middle_slice_tpc, color="#8c05b5", lw=5)

INSET_PENALTY = -6
axin2.set_ylabel("Linear TPC", fontsize=LABEL_FS + INSET_PENALTY)
axin2.set_xlabel("Distance (px)", fontsize=LABEL_FS + INSET_PENALTY)
axin2.xaxis.set_tick_params(labelsize=TICK_FS + INSET_PENALTY)
axin2.yaxis.set_tick_params(labelsize=TICK_FS + INSET_PENALTY)

axin.set_visible(False)
axin2.vlines([end_idx], 0, ymax, ls="--", color="black", lw=3)  # this is r_0
axin2.vlines([cls], 0, ymax, ls="--", color="red", lw=3)


plt.savefig("paper_figures/workflow/img2.png")


bern_fig = plt.figure("bern", dpi=300)
bern_fig.set_size_inches(14, 14)

bern_ax = plt.gca()
bern_ax.imshow(np.zeros_like(img), cmap="binary")
bern_ax.set_axis_off()
bern_ax.set_title(r"Bernoulli Process", fontsize=TITLE_FS)


icls = int(cls)
for y in range(-1 + ih // icls):
    bern_ax.hlines([y * icls], 0, img.shape[1] - icls, color="red", ls="-", lw=1)
    for x in range(-1 + iw // icls):
        subregion = img[y * icls : (y + 1) * icls, x * icls : (x + 1) * icls]
        if np.mean(subregion) > 0.5:
            rect = patches.Rectangle(
                (x * icls, y * icls), icls, icls, fill="red", color="red", alpha=0.65
            )
            bern_ax.add_artist(rect)
        if y == 0:
            bern_ax.vlines(
                [x * icls], 0, img.shape[0] - icls, color="red", ls="-", lw=1
            )
        if x == -2 + iw // icls:
            bern_ax.vlines(
                [(x + 1) * icls], 0, img.shape[0] - icls, color="red", ls="-", lw=1
            )
        if y == -2 + ih // icls:
            bern_ax.hlines(
                [(y + 1) * icls], 0, img.shape[1] - icls, color="red", ls="-", lw=1
            )

bern_ax.text(
    150,
    ih + 70,
    r"Image represents $\|X\| / a_n^2$ samples from",
    fontsize=LABEL_FS + 2,
)
bern_ax.text(
    220,
    ih + 130,
    r"Bernoulli dist. with $p=\Phi_{X}(\omega)$",
    fontsize=LABEL_FS + 2,
)


plt.savefig("paper_figures/workflow/img3.png")

"""


bern_ax.imshow(img, cmap="binary_r")
bern_ax.set_axis_off()


icls = int(cls)
for y in range(int(img.shape[0] / icls) - 1):
    # bern_ax.hlines([y * cls], 0, img.shape[1], color="red", ls="--", lw=1)
    for x in range(int(img.shape[1] / icls) - 1):
        subregion = img[y * icls : (y + 1) * icls, x * icls : (x + 1) * icls]
        if np.mean(subregion) > 0.5:
            rect = patches.Rectangle(
                (x * icls, y * icls), icls, icls, fill="#fc9c00", alpha=0.3
            )
            bern_ax.add_artist(rect)
            # bern_ax.vlines([x * cls], 0, img.shape[0], color="red", ls="--", lw=1)


fig = plt.gcf()
for i in range(2):
    bbox = axs[i].get_position()
    my = bbox.y0 + (bbox.y1 - bbox.y0) / 2
    nx = img.shape[1]
    arrow = patches.Arrow(bbox.x1 + 0.01, my, 0.06, 0, width=0.13, color="#fc9c00")
    fig.add_artist(arrow)

    if i == 0:
        fig.text(bbox.x1 + 0.01 + 0.015, my + 0.03, "FFT", fontsize=20)


plt.show()
"""
