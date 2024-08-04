from representativity import util, core
from porespy.generators import *
import os
import sys
import matplotlib.pyplot as plt
import porespy as ps
from itertools import product
import json

np.random.seed(0)


def factors_to_params(args, im_shape):
    """
    This function is used to convert the factors of the parameters to the actual
    parameters.
    """
    l = np.mean(im_shape)
    size = np.prod(im_shape)
    l_matching = [match for match in args if "_factor_l" in match]
    size_matching = [match for match in args if "_factor_size" in match]
    for match in l_matching:
        arg = match.split("_")[0]
        args[arg] = int(l / args.pop(match))
    for match in size_matching:
        arg = match.split("_")[0]
        args[arg] = int(size / args.pop(match))
    return args


def get_ps_generators():
    """Returns a dictionary with the porespy generators and their arguments."""
    ps_generators = {
        blobs: {
            "blobiness_factor_l": [50, 100, 150, 200, 250],
            "porosity": list(np.arange(0.1, 0.5, 0.1)),
        },
        fractal_noise: {
            "frequency": list(np.arange(0.015, 0.06, 0.01)),
            "octaves": [2, 7, 12],
            "uniform": [True],
            "mode": ["simplex", "value"],
        },
        # voronoi_edges: {'r': [1,2,3], 'ncells_factor_size': np.array([100,1000,10000])}
    }
    return ps_generators


def get_gen_name(generator, value_comb):
    """Returns the name of the generator with the given values."""
    value_comb = list(value_comb)
    for i in range(len(value_comb)):
        value = value_comb[i]
        if isinstance(value, float):
            value_comb[i] = np.round(value, 3)
    value_comb = list(np.array(value_comb).astype(str))
    value_comb = "_".join(value_comb)
    return f"{generator.__name__}_{value_comb}"


def get_ps_gen_names():
    """Returns a list with the names of the porespy generators."""
    ps_generators = get_ps_generators()
    gen_names = []
    for generator, params in ps_generators.items():
        for value_comb in product(*params.values()):
            gen_names.append(get_gen_name(generator, value_comb))
    return gen_names


def json_validation_preprocessing():
    """
    This function is used to prepare the json file for the validation
    """

    # Load the statistics file
    json_validation_path = "representativity/validation/validation.json"
    if os.path.exists(json_validation_path):
        with open(json_validation_path, "r") as file:
            all_data = json.load(file)
    else:  # Create the file if it does not exist
        with open(json_validation_path, "w") as file:
            json.dump(dict(), file)
        with open(json_validation_path, "r") as file:
            all_data = json.load(file)

    gen_names = get_ps_gen_names()

    # TODO we need to see that the blobiness_factor is linear and ncells is quadratic or cubic.

    modes = ["2D", "3D"]
    for mode in modes:
        if f"validation_{mode}" not in all_data:
            all_data[f"validation_{mode}"] = {}
        for gen_name in gen_names:
            if gen_name not in all_data[f"validation_{mode}"]:
                all_data[f"validation_{mode}"][gen_name] = {}

    # Large im sizes for stat. analysis:
    all_data["validation_2D"]["large_im_size"] = [10000, 10000]  # TODO make this bigger
    all_data["validation_3D"]["large_im_size"] = [600, 600, 600]

    # Edge lengths for the experimental statistical analysis:
    all_data["validation_2D"]["edge_lengths_fit"] = list(
        range(500, 1000, 20)
    )  # TODO change back to 500
    all_data["validation_3D"]["edge_lengths_fit"] = list(range(350, 450, 10))

    # Edge lengths for the predicted integral range:
    all_data["validation_2D"]["edge_lengths_pred"] = [600, 800, 1000, 1200, 1400]
    all_data["validation_3D"]["edge_lengths_pred"] = [300, 350, 400]

    return all_data


def in_the_bounds(pf, error, true_pf):
    bounds = [(1 - error) * pf, (1 + error) * pf]
    res = int(true_pf >= bounds[0] and true_pf <= bounds[1])
    return bounds, res

def get_large_im_stack(generator, large_shape, large_im_repeats, args):
    large_ims = []
    for _ in range(large_im_repeats):
        large_im = generator(shape=large_shape, **args)
        
        if generator == blobs:
            if len(large_im.shape) == 2:
                large_im = large_im[2:-2, 2:-2]
            else:
                large_im = large_im[2:-2, 2:-2, 2:-2]
        large_ims.append(large_im)
        plt.plot([large_im[i,:].mean() for i in range(large_im.shape[0])])
        plt.ylabel('Phase fraction')
        plt.xlabel('slice')
        plt.show()
    res = np.stack(large_ims, axis=0)
    if generator == fractal_noise:
        porosity = 0.5
        res = res < porosity
    return res


def ps_error_prediction(dim, data, confidence, error_target):
    ps_generators = get_ps_generators()
    errs = []
    true_clss = []
    clss = []
    one_im_clss = []
    large_shape = data[f"validation_{dim}"]["large_im_size"]
    large_im_repeats = 3 if dim=="2D" else 10
    in_the_bounds_one_im = []
    in_the_bounds_w_model = []
    in_the_bounds_wo_model = []
    iters = 0
    for generator, params in ps_generators.items():
        for value_comb in product(*params.values()):
            args = {key: value for key, value in zip(params.keys(), value_comb)}
            gen_name = get_gen_name(generator, value_comb)
            args = factors_to_params(args, im_shape=large_shape)
            large_im_stack = get_large_im_stack(
                generator, large_shape, large_im_repeats, args
            )
            if generator == blobs:
                cur_large_shape = np.array(large_shape) - 4
            else:
                cur_large_shape = large_shape
            true_pf = np.mean(large_im_stack)
            edge_lengths_fit = data[f"validation_{dim}"]["edge_lengths_fit"]
            true_cls = util.stat_analysis_error(
                large_im_stack, true_pf, edge_lengths_fit
            )
            print(f"Generator {gen_name} with {args}:")
            print(f"True cls: {true_cls}")
            data[f"validation_{dim}"][gen_name]["true_cls"] = true_cls
            data[f"validation_{dim}"][gen_name]["true_pf"] = true_pf
            edge_lengths_pred = data[f"validation_{dim}"]["edge_lengths_pred"]
            for edge_length in edge_lengths_pred:
                edge_lengths_repeats = 40 if dim == "2D" else 4
                for _ in range(edge_lengths_repeats):
                    run_dict = {"edge_length": edge_length}
                    true_error = util.bernouli_from_cls(
                        true_cls, true_pf, [edge_length] * int(dim[0])
                    )
                    first_index = np.random.randint(0, large_im_stack.shape[0])
                    start_idx = [
                        np.random.randint(0, cur_large_shape[i] - edge_length)
                        for i in range(int(dim[0]))
                    ]
                    end_idx = [start_idx[i] + edge_length for i in range(int(dim[0]))]
                    if dim == "2D":
                        small_im = large_im_stack[first_index][
                            start_idx[0] : end_idx[0], start_idx[1] : end_idx[1]
                        ]
                    else:
                        small_im = large_im_stack[first_index][
                            start_idx[0] : end_idx[0],
                            start_idx[1] : end_idx[1],
                            start_idx[2] : end_idx[2],
                        ]
                    
                    print(f'small im shape: {small_im.shape}')
                    # np.save(f'./small_im_{gen_name}_{args}_{edge_length}.npy', small_im)
                    small_im_pf = np.mean(small_im)
                    run_dict["pf"] = small_im_pf
                    one_im_stat_analysis_cls = core.stat_analysis_error_classic(
                        small_im, np.mean(small_im)
                    )
                    one_im_sa_error = util.bernouli_from_cls(
                        one_im_stat_analysis_cls, small_im_pf, [edge_length] * int(dim[0])
                    )
                    print(f"one im error: {one_im_sa_error[0]:.2f}")
                    one_im_sa_error /= 100
                    bounds, in_bounds = in_the_bounds(small_im_pf, one_im_sa_error, true_pf)
                    run_dict["in_bounds_one_im"] = in_bounds
                    run_dict["error_one_im"] = one_im_sa_error[0]
                    in_the_bounds_one_im.append(in_bounds)
                    print(f'current right percentage one im: {np.mean(in_the_bounds_one_im)}')
                    run_dict["one_im_sa_cls"] = one_im_stat_analysis_cls
                    print(f"One image stat analysis cls: {one_im_stat_analysis_cls}")
                    
                    print(f"one im bounds: {bounds}")
                    one_im_clss.append(one_im_stat_analysis_cls)
                    iters += 1
                    for i in range(2):
                        print(f"Iterations: {iters}")

                        with_model = i == 0

                        out = core.make_error_prediction(
                            small_im,
                            confidence=confidence,
                            target_error=error_target,
                            model_error=with_model,
                        )
                        im_err, l_for_err_target, cls = (
                            out["percent_err"],
                            out["l"],
                            out["integral_range"],
                        )
                        true_clss.append(true_cls)
                        clss.append(cls)
                        bounds, in_bounds = in_the_bounds(small_im_pf, im_err, true_pf)
                        if with_model:
                            in_the_bounds_w_model.append(in_bounds)
                            run_dict["model_in_bounds"] = in_bounds
                        else:
                            in_the_bounds_wo_model.append(in_bounds)
                            run_dict["model_wo_gmm_in_bounds"] = in_bounds
                        print(f"Bounds: {bounds}")
                        print(f"True PF: {true_pf}")
                        if with_model:
                            print("With model:")
                            print(
                                f"current right percentage: {np.mean(in_the_bounds_w_model)}"
                            )
                            run_dict["pred_cls"] = cls
                            run_dict["error_w_gmm"] = im_err
                            
                        else:
                            print("Without model:")
                            print(
                                f"current right percentage: {np.mean(in_the_bounds_wo_model)}"
                            )
                            run_dict["error_wo_gmm"] = im_err
                        print(f"edge_length {edge_length}:")
                        print(f"cls: {cls}")
                        
                        print(f"true error: {true_error[0]:.2f}")
                        print(f"error: {im_err*100:.2f}\n")
                        print(f"Length for error target: {l_for_err_target}")
                    
                    data[f"validation_{dim}"][gen_name][f"run_{iters}"] = run_dict
                    print("\n")
            with open("representativity/validation/validation.json", "w") as file:
                json.dump(data, file)

            # plt.imshow(im[150:350,150:350])
            # plt.title(f'{generator.__name__} with {args}')
            # print(f'Error: {100*im_err:.2f} %')
            # print(f'Length for error target: {l_for_err_target}')
            # print(f'CLS: {cls}')
            # clss.append(cls)
            # errs.append(im_err)
            # plt.show()
            # plt.close()
    return errs, true_clss, clss, one_im_clss


if __name__ == "__main__":
    # shape = [1000, 1000]
    all_data = json_validation_preprocessing()
    dim = "3D"
    # get porespy generators:
    errs, true_clss, clss, one_im_clss = ps_error_prediction(
        dim, all_data, confidence=0.95, error_target=0.05
    )
    # plt.scatter(true_clss, clss, label='CLS')
    # plt.scatter(true_clss, one_im_clss, label='One image stat analysis')
    # max_value = max(max(true_clss), max(clss), max(one_im_clss))
    # plt.plot([0, max_value], [0, max_value], 'k--')
    # plt.xlabel('True CLS')
    # plt.ylabel('CLS')
    # plt.legend()
    # plt.show()
