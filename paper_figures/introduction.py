import numpy as np
import matplotlib.pyplot as plt
import tifffile
import random
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import matplotlib.patches as patches
from paper_figures.pred_vs_true_cls import create_tpc_plot
from representativity import core
from paper_figures.model_accuracy import get_prediction_interval_stats
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.lines as mlines
import tifffile


COLOR_INSET = "darkorange"
COLOR_PHI = "blue"
COLOR_IN = "green"
COLOR_OUT = "red"
LINE_W = 1.5

# Plot the data
if __name__ == '__main__':

    dims = ["2D", "3D"]
    
    col_width = 19
    fig = plt.figure(figsize=(col_width, col_width/1.9))
    gs = GridSpec(2, 4)
    # Have some space between the subplots:
    wspace, hspace = 0.24, 0.48
    gs.update(wspace=wspace, hspace=hspace)

    
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
    x_move, y_move = 480, 290
    x1, x2, y1, y2 = x1 + x_move, x2 + x_move, y1 + y_move, y2 + y_move
    sofc_small_im = sofc_large_im[y1:y2, x1:x2]
    ax_sofc_im = fig.add_subplot(gs[0, 0])
    ax_sofc_im.imshow(sofc_large_im, cmap='gray', interpolation='nearest')
    ax_sofc_im.set_xlabel(f"Material's unknown phase fraction: {sofc_large_im.mean():.3f}", fontsize=11)

    patch_positions = []
    phase_fractions = []
    n_repeats = 100000
    # Calculate the phase fractions of random patches:
    for _ in range(n_repeats):
        x1_patch = random.randint(0, middle_indices[0]-small_im_size)
        x2_patch = x1_patch + small_im_size
        y1_patch = random.randint(0, middle_indices[1]-small_im_size)
        y2_patch = y1_patch + small_im_size
        patch_positions.append((x1_patch, x2_patch, y1_patch, y2_patch))
        phase_fractions.append(sofc_large_im[x1_patch:x2_patch, y1_patch:y2_patch].mean())

    num_patches = 6

    for i, (x1_p, x2_p, y1_p, y2_p) in enumerate(patch_positions[:num_patches]):
        ax_sofc_im.add_patch(patches.Rectangle(
            (x1_p, y1_p), small_im_size, small_im_size, edgecolor=COLOR_IN, linewidth=LINE_W+0.2, facecolor='none'))

    # Create the inset:
    inset_shift = 1 + wspace
    ax_inset = ax_sofc_im.inset_axes([inset_shift, 0, 1, 1], xlim=(x1, x2), ylim=(y1, y2))
    ax_inset.set_xlabel(f"Sample's phase fraction: {sofc_small_im.mean():.3f}", fontsize=11)
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
    ax_sofc_im.set_title("(a)", fontsize=12)
    ax_inset.set_title("(b)", fontsize=12)

    

    # Plot the histogram of the phase fractions:
    unknown_dist = fig.add_subplot(gs[0, 2])
    # unknown_dist.hist(phase_fractions, bins=40, color=COLOR_PHI, edgecolor='black', density=True)
    mu, sigma = norm.fit(phase_fractions)
    gap_pfs = (max(phase_fractions) - min(phase_fractions)) / 2 - 0.001
    x = np.linspace(sofc_large_im.mean() - gap_pfs, sofc_large_im.mean() + gap_pfs, 200)
    p = norm.pdf(x, mu, sigma)
    unknown_dist.plot(x, p, color=COLOR_IN, linewidth=2, 
                      label="Same-sized samples unknown\nphase fraction distribution.")
    # Make the plot 1.5 times lower in the plot:
    unknown_dist.set_ylim(0, unknown_dist.get_ylim()[1]*1.8)
    unknown_dist.set_xlim(sofc_large_im.mean() - gap_pfs, sofc_large_im.mean() + gap_pfs)
    unknown_dist.set_yticks([])

    # Now add the unknown material's phase fraction, until the normal distribution:
    ymax = norm.pdf(sofc_large_im.mean(), mu, sigma)
    ymax_mean = ymax / unknown_dist.get_ylim()[1]
    # And the sample phase fraction:
    ymax_sample = norm.pdf(sofc_small_im.mean(), mu, sigma)
    ymax_sample = ymax_sample / unknown_dist.get_ylim()[1] 
    unknown_dist.axvline(sofc_small_im.mean(), ymax=ymax_sample, color=COLOR_INSET, linestyle='--', linewidth=2, 
                         label="Sample's phase fraction")
    unknown_dist.axvline(sofc_large_im.mean(), ymax=ymax_mean, color='black', linestyle='--', linewidth=2,
                         label="Material's unknown phase\nfraction")
    
    unknown_dist.legend(loc='upper left', fontsize=11)
    unknown_dist.set_title("(c)", fontsize=12)
    unknown_dist.set_xlabel('Phase fraction', fontsize=11)
    

    # Rep. calculation:
    same_small_im = fig.add_subplot(gs[1, 0])
    same_small_im.imshow(sofc_small_im, cmap='gray', interpolation='nearest')
    same_small_im.set_xticks([])
    same_small_im.set_yticks([])
    same_small_im.set_xlabel(f"Single image input", fontsize=11)
    for spine in same_small_im.spines.values():
        spine.set_edgecolor(COLOR_INSET)
        spine.set_linewidth(LINE_W)
    same_small_im.set_title('(e)', fontsize=12)

    # Create the TPC plot:
    tpc_plot = fig.add_subplot(gs[1, 1])
    create_tpc_plot(fig, sofc_small_im, 40, {"pred": "g"}, sofc_small_im.mean(), tpc_plot, with_omega_notation=False)
    tpc_plot.set_xlabel('')
    tpc_plot.set_ylabel('')
    tpc_plot.set_xticks([])
    tpc_plot.set_yticks([])
    tpc_plot.set_title("(f)", fontsize=12)

    # Text of representativity analysis:
    rep_text = fig.add_subplot(gs[1, 2])

    microlib_examples = tifffile.imread("paper_figures/figure_data/microlib_examples.tif")
    microlib_examples_size = min(microlib_examples.shape)
    microlib_examples = microlib_examples[-microlib_examples_size:, -microlib_examples_size:]
    microlib_examples = microlib_examples / microlib_examples.max()
    width_mult, height_mult = 0.8, 0.8
    large_microlib_im = np.ones((int(microlib_examples_size / width_mult), int(microlib_examples_size / height_mult)))
    start_idx = int((large_microlib_im.shape[0] - microlib_examples_size) / 2)
    large_microlib_im[-microlib_examples_size:, start_idx:start_idx+microlib_examples_size] = microlib_examples
    rep_text.imshow(large_microlib_im, cmap='gray', interpolation='nearest')

    rep_text.set_title("(g)", fontsize=12)
    text_y = 0.1
    text_pos = (0.5*large_microlib_im.shape[1], text_y*large_microlib_im.shape[0])
    rep_text.text(*text_pos, 
                  "TPC integration and data-driven \ncorrection using MicroLib ", va='center', ha='center')
    font_size = rep_text.texts[0].get_fontsize()
    rep_text.texts[0].set_fontsize(font_size + 3)
    # delete the axis:
    rep_text.axis('off')

    

    
    # Results:

    
    # Create the prediction interval plot:
    res_1_pred = fig.add_subplot(gs[0, 3])
    conf_bounds, pf_1d, cum_sum_sum = get_prediction_interval_stats(sofc_small_im)
    # cumulative sum to the original data:
    original_data = np.diff(cum_sum_sum)
    res_1_pred.plot(pf_1d[1:], original_data, label="Predicted likelihood of\nmaterial's phase fraction", linewidth=2)
    res_1_pred.set_ylim(0, res_1_pred.get_ylim()[1]*1.8)
    res_1_pred.set_xlim(pf_1d[0], pf_1d[-1])
    res_1_pred.set_xlabel('Phase fraction', fontsize=11)
    # No y-ticks:
    # res_1_pred.set_yticks([])
    # Fill between confidence bounds:
    conf_start, conf_end = conf_bounds
    # Fill between the confidence bounds under the curve:
    res_1_pred.fill_between(
        pf_1d[1:], 
        original_data, 
        where=(pf_1d[1:] >= conf_start) & (pf_1d[1:] <= conf_end), 
        alpha=0.2,
        )
    # Plot in dashed vertical lines the materials phase fraction and the inset phase fraction:
    
    ymax = np.argmax((pf_1d[1:] - sofc_small_im.mean()) > 0)
    ymax_small = original_data[ymax] / res_1_pred.get_ylim()[1]
    res_1_pred.axvline(
        sofc_small_im.mean(),
        ymax=ymax_small,
        linestyle="--",
        linewidth=2,
        color=COLOR_INSET,
        label="Sample's phase fraction",
    )
    
    phi = sofc_large_im.mean()
    # find phi to change ymax:
    ymax = np.argmax((pf_1d[1:] - phi) > 0)
    ymax_phi = original_data[ymax] / res_1_pred.get_ylim()[1]
    res_1_pred.axvline(
        phi,
        ymax=ymax_phi,
        linestyle="--",
        linewidth=2,
        color='black',
        label="Material's unknown phase\nfraction",
    )
    # No y-ticks:
    res_1_pred.set_yticks([])
    res_1_pred.set_title("(d)", fontsize=12)

    res_1_pred.set_ylim([0, res_1_pred.get_ylim()[1]])
    inset_pf = sofc_small_im.mean()
    xerr = inset_pf - conf_start
    res_1_pred.errorbar(
        sofc_small_im.mean(), 0.0003, xerr=xerr, fmt='o', capsize=6, 
        color=COLOR_INSET, label="95% confidence interval", linewidth=LINE_W, capthick=LINE_W)
    res_1_pred.legend(loc='upper left', fontsize=11)
    
    

    res_2_pred = fig.add_subplot(gs[1, 3])
    target_percetange_error = 1.5
    target_error = target_percetange_error / 100
    result = core.make_error_prediction(sofc_small_im, confidence=0.95,
                                        target_error=target_error)
    l_for_err_target = int(result["l"])
    percent_error = result["percent_err"]
    print(f'current percent error = {percent_error}, target error = {target_error}, new length = {l_for_err_target}')
    print(f'current length = {small_im_size}, target error = {target_error}, new length = {l_for_err_target}')


    # Create a new figure with space around the original image
    small_im_center = np.ones((l_for_err_target, l_for_err_target))
    mid_start_idx = (l_for_err_target - sofc_small_im.shape[0]) // 2
    small_im_center[mid_start_idx:mid_start_idx+sofc_small_im.shape[0], 
                    mid_start_idx:mid_start_idx+sofc_small_im.shape[1]] = sofc_small_im
    res_2_pred.imshow(small_im_center, cmap='gray', interpolation='nearest')

    # Create the diagonal line pattern for the border
    line_spacing = 40  # Spacing between diagonal lines
    line_color = (0.7, 0.7, 0.7)  # Light gray

    # Create diagonal lines
    alpha = 0.3
    linewidth = 3
    for i in range(0, l_for_err_target, line_spacing):
        x_up = np.arange(i, i + l_for_err_target)[:l_for_err_target-i]
        y_up = np.arange(0, l_for_err_target)[:l_for_err_target-i]
        res_2_pred.plot(x_up, y_up, color=line_color, linewidth=linewidth, alpha=alpha)
        if i > 0:
            x_down = np.arange(0, l_for_err_target)[:l_for_err_target-i]
            y_down = np.arange(i, i + l_for_err_target)[:l_for_err_target-i]
            res_2_pred.plot(x_down, y_down, color=line_color, linewidth=linewidth, alpha=alpha)


    # Turn off the axis
    res_2_pred.set_xticks([])
    res_2_pred.set_yticks([])

    label = f"Sample size needed for ±{target_percetange_error}%\ndeviation from Material's phase\nfraction (95% CI), vs. the\ncurrent ±{np.round(percent_error*100, 2)}% CI."
    # Create custom Line2D objects for the spines
    spine_line = mlines.Line2D([], [], color='black', linestyle='--', 
                               linewidth=2, alpha=alpha, 
                               label=label)
    
    inset_lines = res_2_pred.plot([mid_start_idx,mid_start_idx+sofc_small_im.shape[0]], 
                                [mid_start_idx,mid_start_idx], color=COLOR_INSET, 
                                linewidth=1)
    inset_lines = res_2_pred.plot([mid_start_idx,mid_start_idx+sofc_small_im.shape[0]], 
                                [mid_start_idx+sofc_small_im.shape[0],mid_start_idx+sofc_small_im.shape[0]], color=COLOR_INSET, 
                                linewidth=1)
    inset_lines = res_2_pred.plot([mid_start_idx,mid_start_idx], 
                                [mid_start_idx,mid_start_idx+sofc_small_im.shape[0]], color=COLOR_INSET, 
                                linewidth=1)
    inset_lines = res_2_pred.plot([mid_start_idx+sofc_small_im.shape[0],mid_start_idx+sofc_small_im.shape[0]], 
                                [mid_start_idx,mid_start_idx+sofc_small_im.shape[0]], color=COLOR_INSET, 
                                linewidth=1)

    # Add the custom lines to the legend
    res_2_pred.legend(handles=[spine_line], loc='upper left', fontsize=11)

    for spine in res_2_pred.spines.values():
        spine.set_linestyle("--")
        spine.set_linewidth(2)
        spine.set_alpha(alpha)
    
    res_2_pred.set_title("(h)", fontsize=12)

    positions = []
    for i in range(8):
        positions.append(gs[i//4, i%4].get_position(fig))

    arrow_length = 0.03
    arrow_gap = (positions[5].x0 - positions[4].x0 - positions[4].width)/2 - arrow_length/2

    arrow_alpha = 0.4
    simple_arrows_from = [4, 5, 6]
    for start_idx in simple_arrows_from:
        ptB = (positions[start_idx].x0+positions[start_idx].width+arrow_gap,
               positions[start_idx].y0 + positions[start_idx].height / 2)

        ptE = (ptB[0] + arrow_length, ptB[1])
        
        arrow = patches.FancyArrowPatch(
            ptB, ptE, transform=fig.transFigure,fc = COLOR_INSET, arrowstyle='simple', 
            alpha = arrow_alpha,
            mutation_scale = 40.
            )
        # 5. Add patch to list of objects to draw onto the figure
        fig.patches.append(arrow)

    # Now a special arrow for the last one:
    ptB = (positions[6].x0+positions[6].width,
            positions[6].y0 + positions[6].height / 3 * 2)

    ptE = (positions[3].x0, positions[3].y0 - (positions[3].y0 - (positions[7].y0 + positions[7].height))/3)
    
    arrow = patches.FancyArrowPatch(
        ptB, ptE, transform=fig.transFigure,fc = COLOR_INSET, arrowstyle='simple', alpha = arrow_alpha,
        mutation_scale = 40.
        )
    # 5. Add patch to list of objects to draw onto the figure
    fig.patches.append(arrow)

    # Draw bounding boxes around 1. The problem 2. The solution 3. The results

    # Problem:
    gap_up_down = positions[0].y0 - (positions[4].y0 + positions[4].height)
    gap_right_left = positions[1].x0 - (positions[0].x0 + positions[0].width)

    lower_left_corners = [0, 4, 7]
    widths = [[0, 1, 2], [4, 5, 6], [7]]
    heights = [[0], [4], [3, 7]]
    titles = ["The problem of undersampling", "Representativity analysis", "Prediction results"]

    for llc, width, height, title in zip(lower_left_corners, widths, heights, titles):
        lower_left_corner = (positions[llc].x0 - gap_right_left/3, positions[llc].y0 - gap_up_down/3)
        n_width, n_height = len(width), len(height)
        width_rect = sum([positions[i].width for i in width]) + gap_right_left*(n_width - 1) + gap_right_left*2/3
        height_rect = sum([positions[i].height for i in height]) + gap_up_down*(n_height - 1) + gap_up_down*(1/3+1/2)
        rect = plt.Rectangle(
            # (lower-left corner), width, height
            lower_left_corner, width_rect, height_rect, fill=False, color="k", lw=2, ls="--", alpha=0.5,
            zorder=1000, transform=fig.transFigure, figure=fig
            )
        # Insert the title as text in the top middle of the rectangle:
        text_pos = (lower_left_corner[0] + width_rect/2, lower_left_corner[1] + height_rect - gap_up_down/5)
        fig.text(text_pos[0], text_pos[1], title, ha='center', va='center', 
                 fontsize=18, transform=fig.transFigure)

        fig.patches.append(rect)

    # save the pdf fig in high resolution:
    plt.savefig("paper_figures/output/introduction.pdf", dpi=300)
    
    

    
    
    
    

    