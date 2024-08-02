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


def add_inset_zoom(
    xywh: list[int],
    fig_xywh: list[float],
    inset_data: np.ndarray,
    ax,
) -> object:
    # x0, y0, w, h = xywh
    fx, fy, fw, fh = fig_xywh

    x = np.arange(len(inset_data))
    ymin, ymax = np.amin(inset_data), np.amax(inset_data)
    axin = ax.inset_axes(
        fig_xywh, xlim=(x[0], x[-1]), ylim=(ymin, ymax)
    )  # , xlim=(x[0], x[-1]), ylim=(ymin, ymax)
    # axin.set_xticks([])
    # axin.set_yticks([])
    axin.plot(np.arange(len(inset_data)), inset_data)
    axin.set_xlim(x[0], x[-1])
    axin.set_ylim(30, 31)

    ax.indicate_inset_zoom(axin, edgecolor="black", lw=1)
    # axin.set_ylim((y0 + h, y0))

    axin.patch.set_edgecolor("black")

    axin.patch.set_linewidth(1)

    return axin


fig = plt.figure()

# Create a GridSpec with 2 rows and 4 columns
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.5])

# Add the three full-size square plots
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:, 1])
ax3 = fig.add_subplot(gs[:, 2])

# Add the two smaller square plots in the last column
ax4 = fig.add_subplot(gs[0, 3], aspect="equal")
ax5 = fig.add_subplot(gs[1, 3], aspect="equal")

axs = [ax1, ax2, ax3, ax4, ax5]

fig.set_size_inches(14 * 6, 12)
plt.subplots_adjust(wspace=0.6, left=0.1)

DEFAULT_MICROSTRUCTURE = imread("tests/resources/default.tiff")[0]
phases = np.unique(DEFAULT_MICROSTRUCTURE)
SELECTED_PHASE = phases[0]
img = np.where(DEFAULT_MICROSTRUCTURE == SELECTED_PHASE, 1, 0)

image_phase_fraction = np.mean(img)

tpc = core.radial_tpc(img, volumetric=False)
small_imsize = img.shape[0]
center = 40
center_im = small_imsize // 2
tpc_im = tpc[
    center_im - center : center_im + center, center_im - center : center_im + center
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

axs[0].imshow(img, cmap="binary_r")
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_xlabel(r"$X_{l_1}=" + f"{img.shape[1]}$", fontsize=16)
axs[0].set_ylabel(r"$X_{l_2}=" + f"{img.shape[0]}$", fontsize=16)


axs[1].contourf(tpc_im, cmap="plasma", levels=200)
axs[1].set_aspect("equal")
x_ticks = axs[1].get_xticks()[1:-1]
axs[1].set_xticks(x_ticks, np.int64(np.array(x_ticks) - center))
axs[1].set_yticks(x_ticks, np.int64(np.array(x_ticks) - center))


middle_slice_tpc = tpc[center_im, center_im:]
x = np.arange(0, len(middle_slice_tpc))
ymax = np.max(middle_slice_tpc)


titles = [
    r"Binary Microstructure, $\omega$",
    r"Two-Point Correlation, $T_{r, X}(\omega)$",
    "Bernoulli Process",
]
for i in range(3):
    ax = axs[i]
    ax.set_title(titles[i], fontsize=18)


circle_real = plt.Circle((center, center), cls, fill=False, color="red", lw=2, ls="--")
axs[1].text(
    1.3 * cls,
    3.9 * cls,
    r"CLS, $a_n=" + f"{cls:.2f}" + r"$",
    fontsize=16,
    color="red",
)
axs[1].add_artist(circle_real)
axs[1].text(
    2 * cls,
    0.45 * cls,
    r"Decorrelation at",
    fontsize=14,
    color="black",
)
axs[1].text(
    2.5 * cls,
    0.15 * cls,
    r"$r_o=" + f"{end_idx}" + r"$",
    fontsize=14,
    color="black",
)


axs[0].text(
    0,
    1100,
    r"Phase Fraction $\Phi_{X}(\omega)=" + f"{image_phase_fraction:.4f}" + r"$",
    fontsize=16,
)
axs[0].text(
    0,
    1200,
    r"$X = X_{l_1} \times X_{l_2}, \| X \| =" + f"{img.shape[0] * img.shape[1]}" + r"$",
    fontsize=16,
)


ax0, ay0, aw, ah = [-0.1, 0.8, 0.45, 0.2]
axin = axs[1].inset_axes([ax0, ay0, aw, ah])
axin.plot([center, 1.93 * center], [center, center + 1])
# hide the ticks of the linked axes
axin.set_xticks([])
axin.set_yticks([])


mark_inset(axs[1], axin, 2, 1)

axin_pos = axs[1].get_position()
px0, py0, pw, ph = (
    axin_pos.x0,
    axin_pos.y0,
    axin_pos.x1 - axin_pos.x0,
    axin_pos.y1 - axin_pos.y0,
)

abs_pos = [px0 + ax0 * pw, py0 + ay0 * pw, aw * pw, ah * ph]
axin2 = plt.gcf().add_axes(abs_pos)
axin2.set_ylim([0.1, ymax])
axin2.plot(x, middle_slice_tpc, color="#8c05b5")

axin2.set_ylabel("TPC")
axin2.set_xlabel("Distance (px)")

axin.set_visible(False)
axin2.vlines([end_idx], 0, ymax, ls="--", color="black")  # this is r_0
axin2.vlines([cls], 0, ymax, ls="--", color="red")

axs[2].imshow(img, cmap="binary_r")
axs[2].set_axis_off()


icls = int(cls)
for y in range(int(img.shape[0] / icls) - 1):
    # axs[2].hlines([y * cls], 0, img.shape[1], color="red", ls="--", lw=1)
    for x in range(int(img.shape[1] / icls) - 1):
        subregion = img[y * icls : (y + 1) * icls, x * icls : (x + 1) * icls]
        if np.mean(subregion) > 0.5:
            rect = patches.Rectangle(
                (x * icls, y * icls), icls, icls, fill="#fc9c00", alpha=0.3
            )
            axs[2].add_artist(rect)
            # axs[2].vlines([x * cls], 0, img.shape[0], color="red", ls="--", lw=1)


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
