import json
import numpy as np
import matplotlib.pyplot as plt

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
                if dim == "2D":
                    if not gen_name.startswith("anode"):
                        continue
                else:  # 3D
                    if not gen_name.startswith("separator"):
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

def make_data(dim_res, order):
        data = []
        for key in order:
            num_trials, num_in_bounds = dim_res[key]
            row = [
                f"{num_in_bounds}/{num_trials} = {num_in_bounds/num_trials*100:.2f}%", 
                "95%", 
                f"{np.abs(0.95-num_in_bounds/num_trials)*100:.2f}%"
                ]
            data.append(row)
        return data

def make_table(ax_table, in_bounds_res, title):
    res_2d = in_bounds_res["2D"]
    order = ["one_im", "model_wo_gmm", "model_with_gmm"]
    data_2d = make_data(res_2d, order)
    res_3d = in_bounds_res["3D"]
    data_3d = make_data(res_3d, order)
    table_data = data_2d + data_3d
    # plt.figtext(0.415, 0.485, '(b)', ha='center', va='bottom', fontsize=12)
    ax_table.axis('off')
    colWidths = np.array([0.31, 0.14, 0.14])
    colWidths /= colWidths.sum()
    column_labels = ["Material's true phase fraction in the predicted bounds", "Confidence goal", "Absolute error"]
    row_labels1 = ["Classical subdivision method (2D)", "ImageRep only std prediction (2D)", "ImageRep (2D)"]
    row_labels2 = ["Classical subdivision method (3D)", "ImageRep only std prediction (3D)", "ImageRep (3D)"]
    row_labels = row_labels1 + row_labels2
    table1 = ax_table.table(cellText=table_data, colLabels=column_labels, rowLabels=row_labels, loc='center', colWidths=colWidths)
    for key, cell in table1.get_celld().items():
        cell.set_text_props(ha='center', va='center')
    title_len_addition = len(title) * 0.0029
    ax_table.text(-0.195-title_len_addition, .82, title, ha='left', va='top', transform=ax_table.transAxes)
    imagerep_2d_cell = table1[(3, 2)]  # Cell in the bottom-right corner (last row, last column)
    imagerep_2d_cell.set_facecolor('lightgreen')
    imagerep_3d_cell = table1[(6, 2)]  # Cell in the bottom-right corner (last row, last column)
    imagerep_3d_cell.set_facecolor('lightgreen')

if __name__ == '__main__':
    dims = ["2D", "3D"]
    
    # make a fig with 2 subplots, one for each table:
    # Create a figure with 2 subplots
    fig, (ax_table_porespy, ax_table_experimental) = plt.subplots(2, 1, figsize=(12, 5))
    
    # Make the first table for PoreSpy simulated materials
    in_bounds_res = get_in_bounds_results(dims=dims, porespy_bool=True)
    make_table(ax_table_porespy, in_bounds_res, title='PoreSpy simulated materials')
    
    # Make the second table for the experimental materials
    in_bounds_res = get_in_bounds_results(dims=dims, porespy_bool=False)
    make_table(ax_table_experimental, in_bounds_res, title='Experimental materials')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
