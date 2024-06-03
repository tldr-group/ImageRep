
from porespy.generators import *
from representativity import util
import matplotlib.pyplot as plt
import porespy as ps
from itertools import product
import torch


def get_ps_generators(im_len):
    '''Returns a dictionary with the porespy generators and their arguments.'''
    ps_generators = {blobs: {'blobiness': [im_len/100,im_len/150,im_len/200], 'porosity': list(np.arange(0.1,0.6,0.1))},
                 fractal_noise: {'frequency': list(np.arange(0.015,0.05,0.01)), 'octaves': [2,7,12], 'uniform': [True]},
                 voronoi_edges: {'r': [1,2,3], 'ncells': np.array([im_len**2/100,im_len**2/1000,im_len**2/10000]).astype('int')},}
    return ps_generators


if __name__ == '__main__':
    shape = [1000,1000]

    # get porespy generators:
    ps_generators = get_ps_generators(shape[0])
    
    clss = []
    for generator, params in ps_generators.items():
        for value_comb in product(*params.values()):
            args = {key: value for key, value in zip(params.keys(), value_comb)}
            im = generator(shape=shape, **args)
            if generator == fractal_noise:
                porosity = 0.5
                im = im < porosity
            im = torch.tensor(im).float()
            im_err, l_for_err_target, cls = util.make_error_prediction(im, conf=0.95, err_targ=0.05, model_error=True)
            # plt.imshow(im[150:350,150:350])
            # plt.title(f'{generator.__name__} with {args}')
            print(f'Error: {im_err:.2f} %')
            # print(f'Length for error target: {l_for_err_target}')
            print(f'CLS: {cls}')
            clss.append(cls)
            # plt.show()
            # plt.close()
    plt.hist(clss, bins=15)
    plt.show()


