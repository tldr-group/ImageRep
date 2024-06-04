
from porespy.generators import *
from representativity import util
import matplotlib.pyplot as plt
import porespy as ps
from itertools import product
import torch
import json
import os


def get_ps_generators(im_len):
    '''Returns a dictionary with the porespy generators and their arguments.'''
    ps_generators = {blobs: {'blobiness': [im_len/100,im_len/150,im_len/200], 'porosity': list(np.arange(0.1,0.6,0.1))},
                 fractal_noise: {'frequency': list(np.arange(0.015,0.05,0.01)), 'octaves': [2,7,12], 'uniform': [True]},
                 voronoi_edges: {'r': [1,2,3], 'ncells': np.array([im_len**2/100,im_len**2/1000,im_len**2/10000]).astype('int')}}
    return ps_generators

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

    ps_generators = get_ps_generators(1000)
    gen_names = []
    for generator, params in ps_generators.items():
        for value_comb in product(*params.values()):
            value_comb = list(np.round(value_comb, 3).astype(str))
            value_comb = '_'.join(value_comb)
            gen_names.append(f'{generator.__name__}_{value_comb}')

    # TODO make this into a file, and save the name by blobiness_factor and ncells_factor,
    # TODO we need to see that the blobiness_factor is linear and ncells is quadratic or cubic.
    v_names = ['true_pf', 'true_cls']

    modes = ['2D', '3D']
    for mode in modes:
        if f'{mode}_validation' not in all_data:
            all_data[f'{mode}_validation'] = {}
        for v_name in v_names:
            if v_name not in all_data[f'{mode}_validation']:
                all_data[f'{mode}_validation'][v_name] = {}

    # Edge lengths for the experimental statistical analysis:
    all_data['data_gen_2D']['edge_lengths_fit'] = list(range(500, 1000, 20))  # TODO change back to 500
    all_data['data_gen_3D']['edge_lengths_fit'] = list(range(350, 450, 10))

    # Edge lengths for the predicted integral range:
    all_data['data_gen_2D']['edge_lengths_pred'] =  [600, 1000, 1400]
    all_data['data_gen_3D']['edge_lengths_pred'] = [300, 350, 400]

    return all_data, v_names

def ps_error_prediction(shape, confidence, error_target):
    ps_generators = get_ps_generators(shape[0])
    errs = []
    clss = []
    for generator, params in ps_generators.items():
        for value_comb in product(*params.values()):
            args = {key: value for key, value in zip(params.keys(), value_comb)}
            im = generator(shape=shape, **args)
            if generator == fractal_noise:
                porosity = 0.5
                im = im < porosity
            im = torch.tensor(im).float()
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
    all_data, v_names = json_validation_preprocessing()
    # get porespy generators:
    errs, clss = ps_error_prediction(shape, confidence=0.95, error_target=0.05)
    plt.hist(errs, bins=15)
    plt.show()


