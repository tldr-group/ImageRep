import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tifffile

LINE_W = 2.5

def plot_with_inset(im_large, gs_loc, x_s, x_e, y_s, y_e, label_prefix):
    sofc_small_inpainted = im_large[y_s:y_e, x_s:x_e]
    ax_im = fig.add_subplot(gs_loc)
    ax_im.imshow(im_large, cmap='gray', interpolation='nearest')

    # Create the inset:
    inset_shift = 1.2

    ax_inset = ax_im.inset_axes([inset_shift, 0, 1, 1], xlim=(x_s, x_e), ylim=(y_s, y_e))
    ax_inset.imshow(sofc_small_inpainted, cmap='gray', interpolation='nearest', extent=[x_s, x_e, y_s, y_e])
    
    ax_im.indicate_inset_zoom(ax_inset, alpha=1, linewidth=LINE_W, edgecolor='black')
    ax_im.set_xticks([])
    ax_im.set_yticks([])
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_im.set_title(label_prefix + ' image')
    ax_inset.set_title(label_prefix + ' image (inset)')

if __name__ == '__main__':

    # Plot the origianl and inpainted images of the SOFC anode, with inset to both, showing the
    # differences of the inpainting and the goodness of inpainting:

    col_width = 18
    fig = plt.figure(figsize=(col_width, col_width))
    gs = GridSpec(2, 2)
    # Have some space between the subplots:
    gs.update(hspace=0.12)

    # Create the SOFC anode image, with an inset:
    inpainted_dir = 'validation_data/2D'
    origianl_dir = 'paper_figures/figure_data/anode_segmented_tiff_z046.tif'
    sofc_large_inpainted = tifffile.imread(f"{inpainted_dir}/anode_segmented_tiff_z046_inpainted.tif")
    sofc_large_origianl = tifffile.imread(origianl_dir)
    middle_indices = sofc_large_origianl.shape

    small_im_size = 550
    

    # Subregion of the original image:
    x1, x2, y1, y2 = middle_indices[0]//2-small_im_size//2, middle_indices[0]//2+small_im_size//2, middle_indices[1]//2-small_im_size//2,middle_indices[1]//2+small_im_size//2  
    x_move, y_move = 620, -220
    x1, x2, y1, y2 = x1 + x_move, x2 + x_move, y1 + y_move, y2 + y_move

    plot_with_inset(sofc_large_origianl, gs[0, 0], x1, x2, y1, y2, "Original SOFC")
    plot_with_inset(sofc_large_inpainted, gs[1, 0], x1, x2, y1, y2, "Inpainted SOFC")

    plt.savefig('paper_figures/output/SI_inpainting.pdf', format='pdf', dpi=300)