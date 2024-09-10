from representativity.validation import validation
import numpy as np
import matplotlib.pyplot as plt
import json
import porespy as ps
from itertools import product
import tifffile
import random
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

def get_in_bounds_results(dims, porespy_bool=True):
    # Load the data
    validation_data_dir = 'representativity/validation/validation_w_real.json'
    with open(validation_data_dir, "r") as file:
        validation_data = json.load(file)
    # Get the in-bounds results
    in_bound_results = {}
    for dim in dims:
        dim_results = {
            "one_im": [], "model_with_gmm": [], "model_wo_gmm": []}
        for gen_name in validation_data[f"validation_{dim}"].keys():
            # if there are more generators, they need to be added here:
            if porespy_bool:
                if not gen_name.startswith("blob") and not gen_name.startswith("frac"):
                    continue
            else:
                if not gen_name.startswith("anode"):
                    continue
            gen_data = validation_data[f"validation_{dim}"][gen_name]
            for run in gen_data.keys():
                if not run.startswith("run"):
                    continue
                run_data = gen_data[run]
                dim_results["one_im"].append(run_data["in_bounds_one_im"])
                dim_results["model_with_gmm"].append(run_data["model_in_bounds"])
                dim_results["model_wo_gmm"].append(run_data["model_wo_gmm_in_bounds"])
        n_trials = len(dim_results["one_im"])
        for res in dim_results.keys():
            dim_results[res] = (n_trials, np.array(dim_results[res]).sum())
        in_bound_results[dim] = dim_results
    return in_bound_results

# Plot the data
if __name__ == '__main__':

    dims = ["2D", "3D"]
    
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(4, 4, height_ratios=[2, 1.5, 2, 1])

    # Create the SOFC anode image, with an inset:
    sofc_dir = 'validation_data/2D'
    sofc_large_im = tifffile.imread(f"{sofc_dir}/anode_segmented_tiff_z046_inpainted.tif")
    first_phase = sofc_large_im.min()
    sofc_large_im[sofc_large_im != first_phase] = 1
    sofc_large_im = sofc_large_im[:sofc_large_im.shape[0], :sofc_large_im.shape[0]]
    middle_indices = sofc_large_im.shape
    small_im_size = middle_indices[0]//6
    # Subregion of the original image:
    x1, x2, y1, y2 = middle_indices[0]//2-small_im_size//2, middle_indices[0]//2+small_im_size//2, middle_indices[1]//2-small_im_size//2,middle_indices[1]//2+small_im_size//2  
    sofc_small_im = sofc_large_im[x1:x2, y1:y2]
    ax_sofc_im = fig.add_subplot(gs[0, :2])
    ax_sofc_im.imshow(sofc_large_im, cmap='gray', interpolation='nearest')

    # Create the inset:
    ax_inset = ax_sofc_im.inset_axes([1.2, 0, 1, 1], xlim=(x1, x2), ylim=(y1, y2))
    ax_inset.imshow(sofc_small_im, cmap='gray', interpolation='nearest', extent=[x1, x2, y1, y2])
    ax_sofc_im.indicate_inset_zoom(ax_inset, alpha=1, edgecolor="black")
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
        ax_sofc_im.add_patch(patches.Rectangle((x1, y1), patch_size, patch_size, edgecolor='black', facecolor='none'))
    

    pos3 = ax_sofc_im.get_position() # get the original position
    pos4 = [pos3.x0 - 0.31, pos3.y0+0.035, pos3.width+0.1, pos3.height+0.1] 
    ax_sofc_im.set_position(pos4)


    # No, create the plot showing that the real phase fraction lies within
    # the predicted bounds roughly 95% of the time:
    ax_bars = fig.add_subplot(gs[1, :])
    ax_bars.set_title("(a)")

    # obtain the data:
    in_bounds_res = get_in_bounds_results(dims=dims, porespy_bool=False)
    res_2d = in_bounds_res["2D"]

 
    # make the table:
    # first make the data:
    in_bounds_res = get_in_bounds_results(dims=dims, porespy_bool=True)
    res_2d = in_bounds_res["2D"]
    order = ["one_im", "model_wo_gmm", "model_with_gmm"]

    def make_data(dim_res):
        data = []
        for key in order:
            num_trials, num_in_bounds = dim_res[key]
            row = [
                f"{num_trials}", 
                f"{num_in_bounds}/{num_trials} = {num_in_bounds/num_trials*100:.2f}%", 
                "95%", 
                f"{np.abs(0.95-num_in_bounds/num_trials)*100:.2f}%"
                ]
            data.append(row)
        return data
    
    data_2d = make_data(res_2d)
    res_3d = in_bounds_res["3D"]
    data_3d = make_data(res_3d)
    table_data = data_2d + data_3d
    ax_table = fig.add_subplot(gs[2, :])
    plt.figtext(0.415, 0.485, '(b)', ha='center', va='bottom', fontsize=12)
    ax_table.axis('off')
    colWidths = np.array([0.14, 0.4, 0.14, 0.14])
    colWidths /= colWidths.sum()
    column_labels = ["Number of trials", "Material's true phase fraction is in the predicted bounds", "Confidence goal", "Absolute error"]
    row_labels1 = ["Classical subdivision method (2D)", "ImageRep only std prediction (2D)", "ImageRep (2D)"]
    row_labels2 = ["Classical subdivision method (3D)", "ImageRep only std prediction (3D)", "ImageRep (3D)"]
    row_labels = row_labels1 + row_labels2
    table1 = ax_table.table(cellText=table_data, colLabels=column_labels, rowLabels=row_labels, loc='center', colWidths=colWidths)
    for key, cell in table1.get_celld().items():
        cell.set_text_props(ha='center', va='center')
    ax_table.text(-0.23, .77, 'PoreSpy materials', ha='left', va='top', transform=ax_table.transAxes)
    imagerep_2d_cell = table1[(3, 3)]  # Cell in the bottom-right corner (last row, last column)
    imagerep_2d_cell.set_facecolor('lightgreen')
    imagerep_3d_cell = table1[(6, 3)]  # Cell in the bottom-right corner (last row, last column)
    imagerep_3d_cell.set_facecolor('lightgreen')
    
    # Second table for SOFC anode:
    in_bounds_res = get_in_bounds_results(dims=dims, porespy_bool=False)
    res_2d = in_bounds_res["2D"]
    order = ["one_im", "model_wo_gmm", "model_with_gmm"]

    def make_data(dim_res):
        data = []
        for key in order:
            num_trials, num_in_bounds = dim_res[key]
            row = [
                f"{num_trials}", 
                f"{num_in_bounds}/{num_trials} = {num_in_bounds/num_trials*100:.2f}%", 
                "95%", 
                f"{np.abs(0.95-num_in_bounds/num_trials)*100:.2f}%"
                ]
            data.append(row)
        return data
    
    data_2d = make_data(res_2d)
    table_data = data_2d 
    ax_table = fig.add_subplot(gs[3, :])
    plt.figtext(0.415, 0.485, '(b)', ha='center', va='bottom', fontsize=12)
    ax_table.axis('off')
    colWidths = np.array([0.14, 0.4, 0.14, 0.14])
    colWidths /= colWidths.sum()
    column_labels = ["Number of trials", "Material's true phase fraction is in the predicted bounds", "Confidence goal", "Absolute error"]
    row_labels1 = ["Classical subdivision method (2D)", "ImageRep only std prediction (2D)", "ImageRep (2D)"]
    row_labels = row_labels1 
    table1 = ax_table.table(cellText=table_data, colLabels=column_labels, rowLabels=row_labels, loc='center', colWidths=colWidths)
    for key, cell in table1.get_celld().items():
        cell.set_text_props(ha='center', va='center')
    ax_table.text(-0.23, .77, 'SOFC Anode', ha='left', va='top', transform=ax_table.transAxes)
    imagerep_2d_cell = table1[(3, 3)]  # Cell in the bottom-right corner (last row, last column)
    imagerep_2d_cell.set_facecolor('lightgreen')

    # plt.tight_layout()

    plt.savefig("paper_figures/model_accuracy.pdf", format="pdf", bbox_inches='tight', dpi=300)

