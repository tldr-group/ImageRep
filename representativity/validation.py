
from porespy.generators import *
from representativity import util
import matplotlib.pyplot as plt
import porespy as ps
from itertools import product
import torch
import json
import os

def factors_to_params(args, im_shape):
    '''
    This function is used to convert the factors of the parameters to the actual
    parameters.
    '''
    l = np.mean(im_shape)
    size = np.prod(im_shape)
    l_matching = [match for match in args if '_factor_l' in match]
    size_matching = [match for match in args if '_factor_size' in match]
    for match in l_matching:
        arg = match.split('_')[0]
        args[arg] = int(l/args.pop(match))
    for match in size_matching:
        arg = match.split('_')[0]
        args[arg] = int(size/args.pop(match))
    return args

def get_ps_generators():
    '''Returns a dictionary with the porespy generators and their arguments.'''
    ps_generators = {blobs: {'blobiness_factor_l': [100,150,200], 'porosity': list(np.arange(0.1,0.6,0.1))},
                fractal_noise: {'frequency': list(np.arange(0.015,0.05,0.01)), 'octaves': [2,7,12], 'uniform': [True]},
                voronoi_edges: {'r': [1,2,3], 'ncells_factor_size': np.array([100,1000,10000])}}
    return ps_generators

def get_gen_name(generator, value_comb):
    '''Returns the name of the generator with the given values.'''
    value_comb = list(np.round(value_comb, 3).astype(str))
    value_comb = '_'.join(value_comb)
    return f'{generator.__name__}_{value_comb}'

def get_ps_gen_names():
    '''Returns a list with the names of the porespy generators.'''
    ps_generators = get_ps_generators()
    gen_names = []
    for generator, params in ps_generators.items():
        for value_comb in product(*params.values()):
            gen_names.append(get_gen_name(generator, value_comb))
    return gen_names

def json_validation_preprocessing():
    '''
    This function is used to prepare the json file for the validation
    '''

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

    modes = ['2D', '3D']
    for mode in modes:
        if f'validation_{mode}' not in all_data:
            all_data[f'validation_{mode}'] = {}
        for gen_name in gen_names:
            if gen_name not in all_data[f'validation_{mode}']:
                all_data[f'validation_{mode}'][gen_name] = {}
            # for v_name in v_names:
            #     if v_name not in all_data[f'validation_{mode}'][gen_name]:
            #         all_data[f'validation_{mode}'][gen_name][v_name] = {}

    # Large im sizes for stat. analysis:
    all_data['validation_2D']['large_im_size'] = [10000, 10000]  # TODO make this bigger
    all_data['validation_3D']['large_im_size'] = [3000, 3000, 3000]

    # Edge lengths for the experimental statistical analysis:
    all_data['validation_2D']['edge_lengths_fit'] = list(range(500, 1000, 20))  # TODO change back to 500
    all_data['validation_3D']['edge_lengths_fit'] = list(range(350, 450, 10))

    # Edge lengths for the predicted integral range:
    all_data['validation_2D']['edge_lengths_pred'] =  [600, 1000, 1400]
    all_data['validation_3D']['edge_lengths_pred'] = [300, 350, 400]

    return all_data

def get_large_im_stack(generator, large_shape, large_im_repeats, args):
    large_ims = []
    for _ in range(large_im_repeats):
        large_im = generator(shape=large_shape, **args)
        if generator == fractal_noise:
            porosity = 0.5
            large_im = large_im < porosity
        large_im = torch.tensor(large_im).float()
        large_ims.append(large_im)
    return torch.stack(large_ims, axis=0)

def ps_error_prediction(dim, data, confidence, error_target):
    ps_generators = get_ps_generators()
    errs = []
    clss = []
    large_shape = data[f'validation_{dim}']['large_im_size']
    large_im_repeats = 1
    for generator, params in ps_generators.items():
        for value_comb in product(*params.values()):
            args = {key: value for key, value in zip(params.keys(), value_comb)}
            gen_name = get_gen_name(generator, value_comb)
            args = factors_to_params(args, im_shape=large_shape)
            large_im_stack = get_large_im_stack(generator, large_shape, large_im_repeats, args)
            
            im_err, l_for_err_target, cls = util.make_error_prediction(im, 
                conf=confidence, err_targ=error_target, model_error=True)
            # plt.imshow(im[150:350,150:350])
            # plt.title(f'{generator.__name__} with {args}')
            print(f'Error: {100*im_err:.2f} %')
            print(f'Length for error target: {l_for_err_target}')
            print(f'CLS: {cls}')
            clss.append(cls)
            errs.append(im_err)
            # plt.show()
            # plt.close()
    return errs, clss

if __name__ == '__main__':
    shape = [1000,1000]
    all_data = json_validation_preprocessing()
    dim = '2D'
    # get porespy generators:
    errs, clss = ps_error_prediction(dim, all_data, confidence=0.95, error_target=0.05)
    plt.hist(errs, bins=15)
    plt.show()


