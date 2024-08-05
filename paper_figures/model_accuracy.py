from representativity.validation import validation
import numpy as np
import matplotlib.pyplot as plt
import json
import porespy as ps
from itertools import product
import random
from matplotlib.gridspec import GridSpec

def get_in_bounds_results(dims):
    # Load the data
    validation_data_dir = 'representativity/validation/validation.json'
    with open(validation_data_dir, "r") as file:
        validation_data = json.load(file)
    # Get the in-bounds results
    in_bound_results = {}
    for dim in dims:
        dim_results = {
            "one_im": [], "model_with_gmm": [], "model_wo_gmm": []}
        for gen_name in validation_data[f"validation_{dim}"].keys():
            # if there are more generators, they need to be added here:
            if not gen_name.startswith("blob") and not gen_name.startswith("frac"):
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
    
    num_generators = 50
    num_images = 5
    generators_chosen = np.random.choice(num_generators, num_images, replace=False)
    images = []
    large_img_size = np.array([1000, 1000])
    img_size = np.array([200, 200])
    alpha = 1
    ps_generators = validation.get_ps_generators()
    rand_iter = 0
    for generator, params in ps_generators.items():
        for value_comb in product(*params.values()):
            if rand_iter in generators_chosen:
                args = {key: value for key, value in zip(params.keys(), value_comb)}
                args = validation.factors_to_params(args, im_shape=large_img_size)
                image = validation.get_large_im_stack(generator, large_img_size, 1, args)
                image = image[0]
                image = image[:img_size[0], :img_size[1]]
                images.append(image)
            rand_iter += 1
    random.shuffle(images)
    
    layers = num_images  # How many images should be stacked.
    x_offset, y_offset = img_size[0]-25, 30  # Number of pixels to offset each image.

    new_shape = ((layers - 1)*y_offset + images[0].shape[0],
                (layers - 1)*x_offset + images[0].shape[1]
                )  # the last number, i.e. 4, refers to the 4 different channels, being RGB + alpha

    stacked = np.zeros(new_shape)

    for layer in range(layers):
        cur_im = images[layer]
        stacked[layer*y_offset:layer*y_offset + cur_im.shape[0],
                layer*x_offset:layer*x_offset + cur_im.shape[1] 
                ] += images[layer]
    stacked = 1 - stacked

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 1, height_ratios=[1, 2])

    ax_im = fig.add_subplot(gs[0])
    ax_im.imshow(stacked, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
    ax_im.set_title('(a)')
    ax_im.axis('off') 

    pos1 = ax_im.get_position() # get the original position
    pos2 = [pos1.x0 - 0.15, pos1.y0-0.1, pos1.width+0.1, pos1.height+0.1] 
    ax_im.set_position(pos2) 
    # make the table:
    # first make the data:
    in_bounds_res = get_in_bounds_results(dims=dims)
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
    ax_table = fig.add_subplot(gs[1])
    plt.figtext(0.415, 0.485, '(b)', ha='center', va='bottom', fontsize=12)
    ax_table.axis('off')
    colWidths = np.array([0.14, 0.4, 0.14, 0.14])
    colWidths /= colWidths.sum()
    column_labels = ["Number of trials", "True phase fraction in the predicted bounds", "Confidence goal", "Absolute error"]
    row_labels1 = ["Classical subdivision method (2D)", "ImageRep without GMM step (2D)", "ImageRep (2D)"]
    row_labels2 = ["Classical subdivision method (3D)", "ImageRep without GMM step (3D)", "ImageRep (3D)"]
    row_labels = row_labels1 + row_labels2
    table1 = ax_table.table(cellText=table_data, colLabels=column_labels, rowLabels=row_labels, loc='center', colWidths=colWidths)
    for key, cell in table1.get_celld().items():
        cell.set_text_props(ha='center', va='center')
    imagerep_2d_cell = table1[(3, 3)]  # Cell in the bottom-right corner (last row, last column)
    imagerep_2d_cell.set_facecolor('lightgreen')
    imagerep_3d_cell = table1[(6, 3)]  # Cell in the bottom-right corner (last row, last column)
    imagerep_3d_cell.set_facecolor('lightgreen')
    
    plt.savefig("paper_figures/model_accuracy.pdf", format="pdf", bbox_inches='tight')

