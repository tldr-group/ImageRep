from representativity.validation import validation
import numpy as np
import matplotlib.pyplot as plt
import json

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
    print("Plotting the model accuracy")
    print(get_in_bounds_results(dims=["2D", "3D"]))
