from representativity.validation import validation
import numpy as np
import matplotlib.pyplot as plt
import json
import porespy as ps
from itertools import product
import random

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
        for res in dim_results.keys():
            dim_results[res] = np.array(dim_results[res]).mean()
        in_bound_results[dim] = dim_results
    return in_bound_results

# Plot the data
if __name__ == '__main__':

    dims = ["2D", "3D"]
    in_bounds_res = get_in_bounds_results(dims=dims)

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
                # image = np.repeat(image[..., np.newaxis], 4, axis=2)
                # image[...,2] = alpha
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

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))

    ax[0].imshow(stacked, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
    ax[0].set_title('(a)')
    ax[0].axis('off') 

    ax[1].axis('off')
    colWidths = np.array([0.14, 0.4, 0.14, 0.14])
    colWidths /= colWidths.sum()
    column_labels = ["Number of trials", "True phase fraction in the predicted bounds", "Goal", "Accuracy"]
    row_labels1 = ["Classical subdivision method (2D)", "ImageRep without GMM step (2D)", "ImageRep (2D)"]
    row_labels2 = ["Classical subdivision method (3D)", "ImageRep without GMM step (3D)", "ImageRep (3D)"]
    row_labels = row_labels1 + row_labels2
    table_data = np.random.randint(1, 10, size=(6, 4))
    table1 = ax[1].table(cellText=table_data, colLabels=column_labels, rowLabels=row_labels, loc='center', colWidths=colWidths)
    column_labels = ["Number of trials", "True phase fraction in the predicted bounds", "Goal", "Accuracy"]
    for key, cell in table1.get_celld().items():
        cell.set_text_props(ha='center', va='center')

    # row_labels = ["Classical subdivision method (

    # Adjust layout to make room for the table
    # plt.subplots_adjust(left=0.2, top=0.8, bottom=0.1)

    plt.show()

