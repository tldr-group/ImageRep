import numpy as np
import matplotlib.pyplot as plt
import tifffile
import random
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from paper_figures.pred_vs_true_cls import create_tpc_plot
from representativity import core

COLOR_INSET = "darkorange"
COLOR_PHI = "blue"
COLOR_IN = "green"
COLOR_OUT = "red"
LINE_W = 1.5


def get_prediction_interval_stats(inset_image, conf_level=0.95, n_divisions=301):
    phase_fraction = float(np.mean(inset_image))
    n_dims = len(inset_image.shape)  # 2D or 3D
    n_elems = int(np.prod(inset_image.shape))

    two_point_correlation = core.radial_tpc(inset_image, n_dims == 3, True)
    integral_range = core.tpc_to_cls(
        two_point_correlation,
        inset_image,
    )

    n = core.n_samples_from_dims(
        [np.array(inset_image.shape, dtype=np.int32)], integral_range
    )
    # bern = bernouilli
    std_bern = (
        (1 / n[0]) * (phase_fraction * (1 - phase_fraction))
    ) ** 0.5  # this is the std of phi relative to Phi with
    std_model = core.get_std_model(n_dims, n_elems)
    n = core.n_samples_from_dims(
        [np.array(inset_image.shape, dtype=np.int32)], integral_range
    )
    
    std_model = core.get_std_model(n_dims, n_elems)
    conf_bounds, pf_1d, cum_sum_sum = core.get_prediction_interval(
        inset_image.mean(),
        std_bern,
        std_model,
        conf_level=conf_level,
        n_divisions=n_divisions
    )
    
    return conf_bounds, pf_1d, cum_sum_sum

# Plot the data
if __name__ == '__main__':

    dims = ["2D", "3D"]
    
    col_width = 18
    fig = plt.figure(figsize=(col_width, col_width/2.3))
    gs = GridSpec(2, 3, width_ratios=[1.9, 1, 1])
    # Have some space between the subplots:
    gs.update(wspace=0.48, hspace=0.48)

    # Create the SOFC anode image, with an inset:
    sofc_dir = 'validation_data/2D'
    sofc_large_im = tifffile.imread(f"{sofc_dir}/anode_segmented_tiff_z046_inpainted.tif")
    chosen_phase = np.unique(sofc_large_im)[2]
    sofc_large_im[sofc_large_im != chosen_phase] = 7  # a random number 
    sofc_large_im[sofc_large_im == chosen_phase] = 0
    sofc_large_im[sofc_large_im == 7] = 1
    sofc_large_im = sofc_large_im[:sofc_large_im.shape[0], :sofc_large_im.shape[0]]
    middle_indices = sofc_large_im.shape
    small_im_size = 350
    
    # Subregion of the original image:
    x1, x2, y1, y2 = middle_indices[0]//2-small_im_size//2, middle_indices[0]//2+small_im_size//2, middle_indices[1]//2-small_im_size//2,middle_indices[1]//2+small_im_size//2  
    x_move, y_move = 220, -220
    x1, x2, y1, y2 = x1 + x_move, x2 + x_move, y1 + y_move, y2 + y_move
    sofc_small_im = sofc_large_im[x1:x2, y1:y2]
    ax_sofc_im = fig.add_subplot(gs[0, 0])
    ax_sofc_im.imshow(sofc_large_im, cmap='gray', interpolation='nearest')
    ax_sofc_im.set_xlabel(f"Material's phase fraction $\phi$: {sofc_large_im.mean():.3f}")

    # Create the inset:
    inset_shift = 1.2
    ax_inset = ax_sofc_im.inset_axes([inset_shift, 0, 1, 1], xlim=(x1, x2), ylim=(y1, y2))
    ax_inset.set_xlabel(f"Inset phase fraction: {sofc_small_im.mean():.3f}")
    inset_pos = ax_inset.get_position()
    ax_inset.imshow(sofc_small_im, cmap='gray', interpolation='nearest', extent=[x1, x2, y1, y2])
    for spine in ax_inset.spines.values():
        spine.set_edgecolor(COLOR_INSET)
        spine.set_linewidth(LINE_W)
    ax_sofc_im.indicate_inset_zoom(ax_inset, alpha=1, edgecolor=COLOR_INSET, linewidth=LINE_W)
    ax_sofc_im.set_xticks([])
    ax_sofc_im.set_yticks([])
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_sofc_im.set_title("(a)")
    ax_inset.set_title("(b)")
    
    

    pos3 = ax_sofc_im.get_position() # get the original position
    # pos4 = [pos3.x0, pos3.y0, pos3.width, pos3.height] 
    # ax_sofc_im.set_position(pos4)

    ax_bars = fig.add_subplot(gs[1, :])
    pos_ax_bars = ax_bars.get_position()
    pos4 = [pos_ax_bars.x0, pos3.y0, pos3.width, pos3.height] 
    ax_sofc_im.set_position(pos4)

    tpc_plot = fig.add_subplot(gs[0, 1])
    pos5 = tpc_plot.get_position() # get the original position
    

    arrow_gap = 0.015
    arrow_length = 0.032
    # Create an arrow between the right of the inset and left of the FFT plot:
    ptB = (pos4[0]+pos3.width*(1+inset_shift)+arrow_gap, pos4[1] + pos4[3] / 2)
    ptE = (ptB[0] + arrow_length, ptB[1])
    
    arrow = patches.FancyArrowPatch(
        ptB, ptE, transform=fig.transFigure,fc = COLOR_INSET, arrowstyle='simple', alpha = 0.3,
        mutation_scale = 40.
        )
    # 5. Add patch to list of objects to draw onto the figure
    fig.patches.append(arrow)

    # Create the TPC plot:
    circle_pred, cls, cbar = create_tpc_plot(fig, sofc_small_im, 40, {"pred": "g"}, sofc_small_im.mean(), tpc_plot)
    tpc_plot.set_xlabel('TPC distance')
    tpc_plot.set_ylabel('TPC distance')
    tpc_plot.set_title("(c)")
    tpc_plot.legend([circle_pred], [f"Predicted Char. l. s.: {np.round(cls, 2)}"], loc='upper left')
    cbar.set_label(f'TPC function')

    # Create the prediction interval plot:
    pred_interval_ax = fig.add_subplot(gs[0, 2])
    conf_bounds, pf_1d, cum_sum_sum = get_prediction_interval_stats(sofc_small_im)
    # cumulative sum to the original data:
    original_data = np.diff(cum_sum_sum)
    pred_interval_ax.plot(pf_1d[:-1], original_data, label="ImageRep likelihood of $\phi$\ngiven only inset image")

    pred_interval_ax.set_xlabel('Phase fraction')
    # No y-ticks:
    # pred_interval_ax.set_yticks([])
    # Fill between confidence bounds:
    conf_start, conf_end = conf_bounds
    # Fill between the confidence bounds under the curve:
    pred_interval_ax.fill_between(
        pf_1d[:-1], 
        original_data, 
        where=(pf_1d[:-1] >= conf_start) & (pf_1d[:-1] <= conf_end), 
        alpha=0.3
        )
    # Plot in dashed vertical lines the materials phase fraction and the inset phase fraction:
    phi = sofc_large_im.mean()
    pred_interval_ax.vlines(
        phi,
        0,
        np.max(original_data),
        linestyle="--",
        color=COLOR_PHI,
        label="$\phi$",
    )
    pred_interval_ax.vlines(
        sofc_small_im.mean(),
        0,
        np.max(original_data),
        linestyle="--",
        color=COLOR_INSET,
        label="Inset phase fraction",
    )
    
    # No y-ticks:
    pred_interval_ax.set_yticks([])
    pred_interval_ax.set_title("(d)")

    pred_interval_ax.set_ylim([0, pred_interval_ax.get_ylim()[1]])
    inset_pf = sofc_small_im.mean()
    xerr = inset_pf - conf_start
    pred_interval_ax.errorbar(
        sofc_small_im.mean(), 0.0002, xerr=xerr, fmt='o', capsize=6, color=COLOR_INSET, label="95% confidence interval", linewidth=LINE_W)
    pred_interval_ax.legend(loc='upper left')

    # Plot an arrow between the TPC plot and the prediction interval plot:
    ptB_2 = (pos5.x0 + pos5.width + arrow_gap + 0.032, pos4[1] + pos4[3] / 2)
    ptE_2 = (ptB_2[0] + arrow_length, ptB_2[1])

    arrow_2 = patches.FancyArrowPatch(
        ptB_2, ptE_2, transform=fig.transFigure,fc = COLOR_INSET, arrowstyle='simple', alpha = 0.3,
        mutation_scale = 40.
        )
    # Add the arrow:
    fig.patches.append(arrow_2)

    # Now another curly arrow between the prediction interval plot and the model accuracy plot:
    pred_interval_pos = pred_interval_ax.get_position()
    ptB_3 = (pred_interval_pos.x0 + pred_interval_pos.width / 2 + 0.04, pred_interval_pos.y0 - 0.04)
    ptE_3 = (ptB_3[0] - 0.015, ptB_3[1] - 0.2)
    fancy_arrow = patches.FancyArrowPatch(
        ptB_3, ptE_3, transform=fig.transFigure,  # Place arrow in figure coord system
        fc = COLOR_INSET, connectionstyle="arc3,rad=-0.2", arrowstyle='simple', alpha = 0.3,
        mutation_scale = 40.
        )
    fig.patches.append(fancy_arrow)

    # Plot the model accuracy:
    # First, plot \phi as a horizontal line:
    ax_bars.axhline(sofc_large_im.mean(), color=COLOR_PHI, linestyle="--", label="$\phi$")

    # Add some patches of the same size as the inset:
    patch_size = small_im_size
    num_images = 40
    num_patches = 5
    # Randomly place the patches, just not overlapping the center:
    patch_positions = []
    images = []
    for i in range(num_images):
        x1 = random.randint(0, middle_indices[0]-patch_size)
        x2 = x1 + patch_size
        y1 = random.randint(0, middle_indices[1]-patch_size)
        y2 = y1 + patch_size
        patch_positions.append((x1, x2, y1, y2))
        images.append(sofc_large_im[x1:x2, y1:y2])

    images[-2] = sofc_small_im
    # Calculate the confidence bounds for all insets:
    insets_phase_fractions = [im.mean() for im in images]
    conf_errors = []
    for im in images:
        conf_bounds, _, _ = get_prediction_interval_stats(im)
        conf_errors.append(conf_bounds[1] - im.mean())
    
    # For each patch, the ones which \phi is within the confidence bounds are colored green, otherwise red:
    color_inside = []
    for phase_fraction, error in zip(insets_phase_fractions, conf_errors):
        if phase_fraction - error <= phi <= phase_fraction + error:
            color_inside.append(COLOR_IN)
        else:   
            color_inside.append(COLOR_OUT)
    color_inside[-2] = COLOR_INSET  # make the inset image a different color
    in_bool = False
    out_bool = False
    for x, y, error, color in zip(
        np.arange(num_images), insets_phase_fractions, conf_errors, color_inside):
        if color == COLOR_IN:
            if not in_bool:
                ax_bars.errorbar(x, y, yerr=error, lw=2, capsize=3, fmt='o',
                    capthick=2, ls='none', color=color, ecolor=color,
                    label="$\phi$ within predicted 95% CI")
                in_bool = True
            else:
                ax_bars.errorbar(x, y, yerr=error, lw=2, capsize=3, fmt='o',
                    capthick=2, ls='none', color=color, ecolor=color)
        elif color == COLOR_OUT:
            if not out_bool:
                ax_bars.errorbar(x, y, yerr=error, lw=2, capsize=3, fmt='o',
                    capthick=2, ls='none', color=color, ecolor=color,
                    label="$\phi$ outside predicted 95% CI")
                out_bool = True
            else:
                ax_bars.errorbar(x, y, yerr=error, lw=2, capsize=3, fmt='o',
                    capthick=2, ls='none', color=color, ecolor=color)
        else:   
            ax_bars.errorbar(x, y, yerr=error, lw=2, capsize=3, fmt='o',
                capthick=2, ls='none', color=color, ecolor=color,
                )
    ax_bars.legend(loc='upper left')
    
    ax_bars.set_ylabel("Phase fraction")
    ax_bars.set_xlabel("Inset image number")
    # add 1 to the x-ticks:
    ax_bars.set_xticks(np.arange(0, num_images, 5))
    ax_bars.set_xticklabels(np.arange(1, num_images+1, 5))
    ax_bars.set_title("(e)")

    for i, (x1, x2, y1, y2) in enumerate(patch_positions):
        if i > num_patches:
            break
        ax_sofc_im.add_patch(patches.Rectangle(
            (x1, y1), patch_size, patch_size, edgecolor=color_inside[i], linewidth=LINE_W, facecolor='none'))

    plt.savefig("paper_figures/output/model_accuracy.pdf", format="pdf", bbox_inches='tight', dpi=300)

