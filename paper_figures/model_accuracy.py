import numpy as np
import matplotlib.pyplot as plt
import json
import tifffile
import random
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from paper_figures.pred_vs_true_cls import create_tpc_plot

COLOR_INSET = "darkorange"

# Plot the data
if __name__ == '__main__':

    dims = ["2D", "3D"]
    
    col_width = 20
    fig = plt.figure(figsize=(col_width, col_width/3))
    gs = GridSpec(2, 4)

    # Create the SOFC anode image, with an inset:
    sofc_dir = 'validation_data/2D'
    sofc_large_im = tifffile.imread(f"{sofc_dir}/anode_segmented_tiff_z046_inpainted.tif")
    chosen_phase = np.unique(sofc_large_im)[2]
    sofc_large_im[sofc_large_im != chosen_phase] = 7  # a random number 
    sofc_large_im[sofc_large_im == chosen_phase] = 0
    sofc_large_im[sofc_large_im == 7] = 1
    sofc_large_im = sofc_large_im[:sofc_large_im.shape[0], :sofc_large_im.shape[0]]
    middle_indices = sofc_large_im.shape
    small_im_size = middle_indices[0]//6
    
    # Subregion of the original image:
    x1, x2, y1, y2 = middle_indices[0]//2-small_im_size//2, middle_indices[0]//2+small_im_size//2, middle_indices[1]//2-small_im_size//2,middle_indices[1]//2+small_im_size//2  
    x1, x2, y1, y2 = x1 + 100, x2 + 100, y1 - 300, y2 - 300
    sofc_small_im = sofc_large_im[x1:x2, y1:y2]
    ax_sofc_im = fig.add_subplot(gs[0, 0])
    ax_sofc_im.imshow(sofc_large_im, cmap='gray', interpolation='nearest')

    # Create the inset:
    ax_inset = ax_sofc_im.inset_axes([1.2, 0, 1, 1], xlim=(x1, x2), ylim=(y1, y2))
    inset_pos = ax_inset.get_position()
    ax_inset.imshow(sofc_small_im, cmap='gray', interpolation='nearest', extent=[x1, x2, y1, y2])
    for spine in ax_inset.spines.values():
        spine.set_edgecolor(COLOR_INSET)
    ax_sofc_im.indicate_inset_zoom(ax_inset, alpha=1, edgecolor=COLOR_INSET)
    ax_sofc_im.set_xticks([])
    ax_sofc_im.set_yticks([])
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # Add some patches of the same size as the inset:
    patch_size = middle_indices[0]//6
    num_patches = 6
    # Randomly place the patches, just not overlapping the center:
    patch_positions = []
    for i in range(num_patches):
        x1 = random.randint(0, middle_indices[0]-patch_size)
        x2 = x1 + patch_size
        y1 = random.randint(0, middle_indices[1]-patch_size)
        y2 = y1 + patch_size
        patch_positions.append((x1, x2, y1, y2))
    for i, (x1, x2, y1, y2) in enumerate(patch_positions):
        ax_sofc_im.add_patch(patches.Rectangle((x1, y1), patch_size, patch_size, edgecolor=COLOR_INSET, facecolor='none'))
    

    pos3 = ax_sofc_im.get_position() # get the original position
    pos4 = [pos3.x0, pos3.y0, pos3.width, pos3.height] 

    # pos4 = [pos3.x0 - 0.28, pos3.y0, pos3.width, pos3.height] 
    # ax_sofc_im.set_position(pos4)

    tpc_plot = fig.add_subplot(gs[0, 2])
    pos5 = tpc_plot.get_position() # get the original position
    

    arrow_gap = 0.01
    # Create an arrow between the right of the inset and left of the FFT plot:
    ptB = (pos4[0]+pos3.width*2.2+arrow_gap, pos4[1] + pos4[3] / 2)
    ptE = (ptB[0] + 0.05, ptB[1])
    
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
    tpc_plot.legend([circle_pred], [f"Predicted Char. l. s.: {np.round(cls, 2)}"], loc='upper right')
    cbar.set_label(f'TPC function')

    # Bounds plot:
    bounds_plot = fig.add_subplot(gs[0, 3])

    # No, create the plot showing that the real phase fraction lies within
    # the predicted bounds roughly 95% of the time:
    ax_bars = fig.add_subplot(gs[1, :])

    # plt.tight_layout()

    plt.savefig("paper_figures/output/model_accuracy.pdf", format="pdf", bbox_inches='tight', dpi=300)

