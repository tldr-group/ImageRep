import json
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties

def get_in_bounds_results(dims, name="porespy"):
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
            if name == "porespy":
                if not gen_name.startswith("blob") and not gen_name.startswith("frac"):
                    continue
            else:
                if dim == "2D":
                    if not gen_name.startswith("anode"):
                        continue
                else:
                    if name == 'Targray':  # 3D
                        if not gen_name.startswith("separator_Targray"):
                            continue
                    else:
                        if not gen_name.startswith("separator_PP1615"):
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
            dim_results[res] = np.array([n_trials, np.array(dim_results[res]).sum()])
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

def bold_min_value(table_data, table1, start_idx = 0):
    
    absolute_errors = np.array([float(table_data[i][2][:-1]) for i in range(start_idx, start_idx+3)])
    min_indices = np.where(absolute_errors==absolute_errors.min())[0]
    for min_idx in min_indices:
        imagerep_2d_cell_right = table1[(start_idx+min_idx+1, 2)]
        imagerep_2d_cell_right.set_text_props(fontproperties=FontProperties(weight='bold'))
        imagerep_2d_cell_left = table1[(start_idx+min_idx+1, -1)]
        imagerep_2d_cell_left.set_text_props(fontproperties=FontProperties(weight='bold'))
        # imagerep_2d_cell.set_facecolor('lightgreen')

def make_table(dims, ax_table, in_bounds_res, title):
    
    order = ["one_im", "model_wo_gmm", "model_with_gmm"]
    dim_data = [make_data(in_bounds_res[dim], order) for dim in dims]
    table_data = reduce(lambda x, y: x + y, dim_data)
    # plt.figtext(0.415, 0.485, '(b)', ha='center', va='bottom', fontsize=12)
    ax_table.axis('off')
    colWidths = np.array([0.31, 0.14, 0.14])
    colWidths /= colWidths.sum()
    column_labels = ["Material's true phase fraction in the predicted bounds", "Confidence goal", "Absolute error"]

    general_row_labels = ["Classical subdivision method", "ImageRep only std", "ImageRep"]
    dim_row_labels = [[f"{general_row_labels[i]} ({dim})" for i in range(len(general_row_labels))] for dim in dims]
    row_labels = reduce(lambda x, y: x + y, dim_row_labels)
    
    table1 = ax_table.table(cellText=table_data, colLabels=column_labels, rowLabels=row_labels, loc='center', colWidths=colWidths)
    for key, cell in table1.get_celld().items():
        cell.set_text_props(ha='center', va='center')

    title_len_addition = len(title) * 0.003
    y_pos = 0.905 if len(dims) == 1 else 0.885
    ax_table.text(-0.11-title_len_addition, y_pos, title, ha='left', va='top', transform=ax_table.transAxes)
    # Find minimum error and highlight the corresponding cell in bold:
    bold_min_value(table_data, table1)
    if len(dims) > 1:
        bold_min_value(table_data, table1, start_idx=3)

def join_all_data(all_data):
    dims = ["2D", "3D"]
    res = {dim: {key: np.array([0,  0]) for key in all_data[0]["2D"].keys()} for dim in dims}
    for dim in dims:
        # add the reults of the dimension together:
        for i in range(len(all_data)):
            if dim in all_data[i]:
                for key in all_data[i][dim].keys():
                    res[dim][key] += all_data[i][dim][key]
    return res

if __name__ == '__main__':
    dims = ["2D", "3D"]
    
    # make a fig with 2 subplots, one for each table:
    # Create a figure with 2 subplots
    col_width = 16
    fig = plt.figure(figsize=(col_width, col_width/2.1))
    gs = GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 2])

    subplot_args = (
        [["2D", "3D"], "porespy", 'PoreSpy simulated materials'],
        [["2D"], "anode", 'Solid Oxide Fuel Cell anode'],
        [["3D"], "Targray", 'Targray separator'],
        [["3D"], "PP1615", 'PP1615 separator']
    )

    all_data = []
    for i, (dims, name, title) in enumerate(subplot_args):
        in_bounds_res = get_in_bounds_results(dims=dims, name=name)
        all_data.append(in_bounds_res)
        make_table(dims, fig.add_subplot(gs[i]), in_bounds_res, title=title)
    
    # Join all the data together:
    all_data = join_all_data(all_data)
    make_table(["2D", "3D"], fig.add_subplot(gs[4]), all_data, title="All materials")

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure, with high dpi
    plt.savefig("paper_figures/output/SI_validation_tables.pdf", format='pdf', dpi=300)

