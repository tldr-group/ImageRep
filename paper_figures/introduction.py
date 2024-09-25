import numpy as np
import matplotlib.pyplot as plt
import tifffile
import random
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import matplotlib.patches as patches
from paper_figures.pred_vs_true_cls import create_tpc_plot
from representativity import core
from scipy.stats import norm
import matplotlib.mlab as mlab

COLOR_INSET = "darkorange"
COLOR_PHI = "blue"
COLOR_IN = "green"
COLOR_OUT = "red"
LINE_W = 1.5

# Plot the data
if __name__ == '__main__':

    dims = ["2D", "3D"]
    
    col_width = 18
    fig = plt.figure(figsize=(col_width, col_width/2))
    gs = GridSpec(2, 4)
    # Have some space between the subplots:
    gs.update(wspace=0.24, hspace=0.48)

    # Create the SOFC anode image, with an inset:
    sofc_dir = 'validation_data/2D'
    sofc_large_im = tifffile.imread(f"{sofc_dir}/anode_segmented_tiff_z046_inpainted.tif")
    chosen_phase = np.unique(sofc_large_im)[2]
    sofc_large_im[sofc_large_im != chosen_phase] = 7  # a random number 
    sofc_large_im[sofc_large_im == chosen_phase] = 0
    sofc_large_im[sofc_large_im == 7] = 1
    sofc_large_im = sofc_large_im[:sofc_large_im.shape[0], :sofc_large_im.shape[0]]
    middle_indices = sofc_large_im.shape
    small_im_size = 300
    
    # Subregion of the original image:
    x1, x2, y1, y2 = middle_indices[0]//2-small_im_size//2, middle_indices[0]//2+small_im_size//2, middle_indices[1]//2-small_im_size//2,middle_indices[1]//2+small_im_size//2  
    x_move, y_move = 400, 0
    x1, x2, y1, y2 = x1 + x_move, x2 + x_move, y1 + y_move, y2 + y_move
    sofc_small_im = sofc_large_im[x1:x2, y1:y2]
    ax_sofc_im = fig.add_subplot(gs[0, 0])
    ax_sofc_im.imshow(sofc_large_im, cmap='gray', interpolation='nearest')
    ax_sofc_im.set_xlabel(f"Unknown material's phase fraction: {sofc_large_im.mean():.3f}")

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
            (x1_p, y1_p), small_im_size, small_im_size, edgecolor=COLOR_IN, linewidth=LINE_W, facecolor='none'))

    # Create the inset:
    inset_shift = 1.2
    ax_inset = ax_sofc_im.inset_axes([inset_shift, 0, 1, 1], xlim=(x1, x2), ylim=(y1, y2))
    ax_inset.set_xlabel(f"Sample's phase fraction: {sofc_small_im.mean():.3f}")
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

    

    # Plot the histogram of the phase fractions:
    unknown_dist = fig.add_subplot(gs[0, 2])
    # unknown_dist.hist(phase_fractions, bins=40, color=COLOR_PHI, edgecolor='black', density=True)
    mu, sigma = norm.fit(phase_fractions)
    x = np.linspace(sofc_large_im.mean() - 0.03, sofc_large_im.mean() + 0.03, 200)
    p = norm.pdf(x, mu, sigma)
    unknown_dist.plot(x, p, color=COLOR_IN, linewidth=2, 
                      label="Unknown distribution of\nsample phase fractions")
    # Make the plot 1.5 times lower in the plot:
    unknown_dist.set_ylim(0, unknown_dist.get_ylim()[1]*1.6)
    unknown_dist.set_xlim(sofc_large_im.mean() - 0.03, sofc_large_im.mean() + 0.03)
    unknown_dist.set_yticks([])

    # Now add the unknown material's phase fraction, until the normal distribution:
    ymax = norm.pdf(sofc_large_im.mean(), mu, sigma)
    ymax_mean = ymax / unknown_dist.get_ylim()[1]
    print(ymax)
    unknown_dist.axvline(sofc_large_im.mean(), ymax=ymax_mean, color='black', linestyle='--', linewidth=2,
                         label="Unknown material's phase\nfraction")
    # And the sample phase fraction:
    ymax_sample = norm.pdf(sofc_small_im.mean(), mu, sigma)
    ymax_sample = ymax_sample / unknown_dist.get_ylim()[1]
    unknown_dist.axvline(sofc_small_im.mean(), ymax=ymax_sample, color=COLOR_INSET, linestyle='--', linewidth=2, 
                         label="Sample's phase fraction")
    unknown_dist.legend(loc='upper right')
    

    # Rep. calculation:
    same_samll_im = fig.add_subplot(gs[1, 0])
    same_samll_im.imshow(sofc_small_im, cmap='gray', interpolation='nearest')

    # Create the TPC plot:
    tpc_plot = fig.add_subplot(gs[1, 1])
    create_tpc_plot(fig, sofc_small_im, 40, {"pred": "g"}, sofc_small_im.mean(), tpc_plot)

    # Results:

    res_1_pred = fig.add_subplot(gs[0, 3])
    res_2_pred = fig.add_subplot(gs[1, 3])