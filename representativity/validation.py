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
            "porosity": list(np.arange(0.1, 0.6, 0.1)),
        },
        fractal_noise: {
            "frequency": list(np.arange(0.015, 0.05, 0.01)),
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
    if os.path.exists("validation.json"):
        with open("validation.json", "r") as file:
            all_data = json.load(file)
    else:  # Create the file if it does not exist
        with open("validation.json", "w") as file:
            json.dump(dict(), file)
        with open("validation.json", "r") as file:
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
            # for v_name in v_names:
            #     if v_name not in all_data[f'validation_{mode}'][gen_name]:
            #         all_data[f'validation_{mode}'][gen_name][v_name] = {}

    # Large im sizes for stat. analysis:
    # all_data["validation_2D"]["large_im_size"] = [10000, 10000]  # TODO make this bigger
    all_data["validation_2D"]["large_im_size"] = [10000, 10000]  # TODO make this bigger
    all_data["validation_3D"]["large_im_size"] = [3000, 3000, 3000]

    # Edge lengths for the experimental statistical analysis:
    all_data["validation_2D"]["edge_lengths_fit"] = list(
        range(500, 1000, 20)
    )  # TODO change back to 500
    all_data["validation_3D"]["edge_lengths_fit"] = list(range(350, 450, 10))

    # Edge lengths for the predicted integral range:
    all_data["validation_2D"]["edge_lengths_pred"] = [600, 800, 1000, 1200, 1400]
    all_data["validation_3D"]["edge_lengths_pred"] = [300, 350, 400]

    return all_data


def get_large_im_stack(generator, large_shape, large_im_repeats, args):
    large_ims = []
    for _ in range(large_im_repeats):
        large_im = generator(shape=large_shape, **args)
        if generator == fractal_noise:
            porosity = 0.5
            large_im = large_im < porosity
        large_im = large_im
        large_ims.append(large_im)
    return np.stack(large_ims, axis=0)


def ps_error_prediction(dim, data, confidence, error_target):
    ps_generators = get_ps_generators()
    errs = []
    true_clss = []
    clss = []
    one_im_clss = []
    large_shape = data[f"validation_{dim}"]["large_im_size"]
    large_im_repeats = 1
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
            true_pf = np.mean(large_im_stack)
            edge_lengths_fit = data[f"validation_{dim}"]["edge_lengths_fit"]
            true_cls = util.stat_analysis_error(
                large_im_stack, true_pf, edge_lengths_fit
            )
            print(f"Generator {gen_name} with {args}:")
            print(f"True cls: {true_cls}")
            data[f"validation_{dim}"][gen_name]["true_cls"] = true_cls
            edge_lengths_pred = data[f"validation_{dim}"]["edge_lengths_pred"]
            for edge_length in edge_lengths_pred:
                for _ in range(20):
                    true_error = util.bernouli_from_cls(
                        true_cls, true_pf, [edge_length] * int(dim[0])
                    )
                    start_idx = [
                        np.random.randint(0, large_shape[i] - edge_length)
                        for i in range(int(dim[0]))
                    ]
                    end_idx = [start_idx[i] + edge_length for i in range(int(dim[0]))]
                    small_im = large_im_stack[0][
                        start_idx[0] : end_idx[0], start_idx[1] : end_idx[1]
                    ]
                    # np.save(f'./small_im_{gen_name}_{args}_{edge_length}.npy', small_im)
                    small_im_pf = np.mean(small_im)
                    one_im_stat_analysis_cls = core.stat_analysis_error_classic(
                        small_im, np.mean(small_im)
                    )
                    print(f"One image stat analysis cls: {one_im_stat_analysis_cls}")
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
                        bounds = [
                            (1 - im_err) * small_im_pf,
                            (1 + im_err) * small_im_pf,
                        ]
                        print(f"Bounds: {bounds}")
                        print(f"True PF: {true_pf}")
                        if true_pf >= bounds[0] and true_pf <= bounds[1]:
                            if with_model:
                                in_the_bounds_w_model.append(1)
                            else:
                                in_the_bounds_wo_model.append(1)
                        else:
                            if with_model:
                                in_the_bounds_w_model.append(0)
                            else:
                                in_the_bounds_wo_model.append(0)
                        if with_model:
                            print("With model:")
                            print(
                                f"current right percentage: {np.mean(in_the_bounds_w_model)}"
                            )
                        else:
                            print("Without model:")
                            print(
                                f"current right percentage: {np.mean(in_the_bounds_wo_model)}"
                            )
                        print(f"edge_length {edge_length}:")
                        print(f"cls: {cls}")
                        print(f"true error: {true_error[0]:.2f}")
                        print(f"error: {im_err*100:.2f}\n")
                        print(f"Length for error target: {l_for_err_target}")
                    if (in_the_bounds_wo_model[-1] == 1) and (
                        in_the_bounds_w_model[-1] == 0
                    ):
                        print("The model is not working properly. Exiting...")
                        sys.exit()
                    print("\n")

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
    shape = [1000, 1000]
    all_data = json_validation_preprocessing()
    dim = "2D"
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
