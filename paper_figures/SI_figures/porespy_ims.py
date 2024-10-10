import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import product
import porespy as ps
from representativity.validation import validation


if __name__ == '__main__':
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
                ] += cur_im
    stacked = 1 - stacked

    

    # Create the PoreSpy images:
    ax_porespy_im = fig.add_subplot(gs[0, 0])
    ax_porespy_im.imshow(stacked, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
    ax_porespy_im.set_title('(a)')
    ax_porespy_im.axis('off') 

    pos1 = ax_porespy_im.get_position() # get the original position
    pos2 = [pos1.x0 - 0.15, pos1.y0+0, pos1.width+0.1, pos1.height+0.1] 
    ax_porespy_im.set_position(pos2) 
