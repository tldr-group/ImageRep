import numpy as np
import torch
import slicegan
from scipy.optimize import minimize
from scipy import stats
from matplotlib import pyplot as plt
from itertools import product
from itertools import chain
import time
from scipy import ndimage

def load_generator(Project_path):
    img_size, img_channels, scale_factor = 64, 1, 1
    z_channels = 16
    lays = 6
    dk, gk = [4] * lays, [4] * lays
    ds, gs = [2] * lays, [2] * lays
    df, gf = [img_channels, 64, 128, 256, 512, 1], [
        z_channels,
        512,
        256,
        128,
        64,
        img_channels,
    ]
    dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

    ## Create Networks
    netD, netG = slicegan.networks.slicegan_nets(
        Project_path, False, "grayscale", dk, ds, df, dp, gk, gs, gf, gp
    )
    netG = netG()
    netG = netG.cuda()
    return netG

def generate_image(netG, slice_dim=0, lf=50, threed=False, reps=50):
    
    netG.eval()
    imgs = []
    
    for _ in range(reps):
        if (_ % 50) == 0 and _ != 0:
            print(f'generating image {_}')
        noise = torch.randn(1, 16, lf if threed else 4, lf, lf)
        noise.transpose_(2, slice_dim+2)
        noise = noise.cuda()
        img = netG(noise, threed, slice_dim)
        img = slicegan.util.post_proc(img)
        img.transpose_(0, slice_dim)
        if not threed:
            imgs.append(img[0])
        else:
            imgs.append(img.cpu())
    img = torch.stack(imgs, 0)
    return img.float()

def angular_img(img):
    base_len, l = img.shape[0:2]
    img = img.cpu().numpy()
    plt.imshow(img[0, :100, :100])
    plt.show()
    img_rot = ndimage.rotate(img, base_len/l*90, axes=(1, 0), reshape=False)
    for i in range(img_rot.shape[0]):
        print(f'slice {i}')
        plt.imshow(img_rot[i, :100, :100])
        plt.show()
        plt.imshow(img_rot[i, -100:, -100:])
        plt.show()
    return img_rot


def tpc_radial(img, mx=100, threed=False, periodic=True):
    desired_length = img.shape[0]//2
    if not periodic:
        desired_length = img.shape[0]-1
    return two_point_correlation(img, desired_length=desired_length, periodic=periodic, threed=threed)


def stat_analysis_error_classic(img, pf):  # TODO see if to delete this or not
    ratios = [2**i for i in np.arange(1, int(np.log2(img.shape[1]))-5)]
    ratios.reverse()
    if img.shape[0] > 1:
        ratios.append(1)
    ratios = ratios[-4:]
    edge_lengths = [img.shape[1]//r for r in ratios]
    img_dims = [np.array((l,)*(len(img.shape)-1)) for l in edge_lengths]
    err_exp = image_stats(img, pf, ratios)
    real_cls = fit_cls(err_exp, img_dims, pf)
    # TODO different size image 1000 vs 1500
    return real_cls


def stat_analysis_error(img, pf, edge_lengths):  # TODO see if to delete this or not
    img_dims = [np.array((l,)*(len(img.shape)-1)) for l in edge_lengths]
    err_exp = real_image_stats(img, edge_lengths, pf)
    real_cls = fit_cls(err_exp, img_dims, pf)
    # TODO different size image 1000 vs 1500
    return real_cls


def real_image_stats(img, ls, pf, repeats=4000, conf=0.95):  
    '''Calculates the error of the stat. analysis for different edge lengths.
    The error is calculated by the std of the mean of the subimages divided by the pf.
    params:
    img: the image to calculate the error for (Should be a stack of images).
    ls: the edge lengths to calculate the error for.
    pf: the phase fraction of the image.
    repeats: the number of repeats for each edge length.
    conf: the confidence level for the error.'''
    dims = len(img[0].shape)
    errs = []
    for l in ls:
        pfs = []
        n_pos_ims = int(np.prod(img.shape)/l**dims)
        repeats = n_pos_ims*2
        # print(f'one im repeats = {repeats} for l = {l}')
        if dims == 1:
            for _ in range(repeats):
                bm, xm = img.shape
                x = torch.randint(0, xm - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l]
                pfs.append(torch.mean(crop).cpu())
        elif dims == 2:
            for _ in range(repeats):
                bm, xm, ym = img.shape
                x = torch.randint(0, xm - l, (1,))
                y = torch.randint(0, ym - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l, y : y + l]
                pfs.append(torch.mean(crop).cpu())
        else:  # 3D
            for _ in range(repeats):
                bm, xm, ym, zm = img.shape
                x = torch.randint(0, xm - l, (1,))
                y = torch.randint(0, ym - l, (1,))
                z = torch.randint(0, zm - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l, y : y + l, z : z + l]
                pfs.append(torch.mean(crop).cpu())
        pfs = np.array(pfs)
        ddof = 1  # for unbiased std
        std = np.std(pfs, ddof=ddof)
        errs.append(100 * (stats.norm.interval(conf, scale=std)[1] / pf))
    return errs

def bernouli_from_cls(cls, pf, img_size, conf=0.95):
    ns = ns_from_dims([np.array(img_size)], cls)
    return bernouli(pf, ns, conf)

def bernouli(pf, ns, conf=0.95):
    errs = []
    for n in ns:
        std_theo = ((1 / n) * (pf * (1 - pf))) ** 0.5
        errs.append(100 * (stats.norm.interval(conf, scale=std_theo)[1] / pf))
    return np.array(errs, dtype=np.float64)


def fit_cls(err_exp, img_dims, pf, max_cls=150):
    err_exp = np.array(err_exp)
    cls = test_cls_set(err_exp, pf, np.arange(1, max_cls, 1), img_dims)
    cls = test_cls_set(err_exp, pf, np.linspace(cls - 1, cls + 1, 50), img_dims)
    # print(f'real cls = {cls}')
    return cls


def ns_from_dims(img_dims, cls):
    n_dims = len(img_dims[0])
    den = cls ** n_dims
    # return [np.prod(np.array(i)) / den for i in img_dims]
    return [np.prod(np.array(i)) / den for i in img_dims]
    # if n_dims == 3:  # 2cls length
    #     return [np.prod(i + 2*(cls - 1)) / den for i in img_dims]
    # else:  # n_dims == 2
    #     return [np.prod(i + cls - 1) / den for i in img_dims]

def dims_from_n(n, shape, cls, dims):
    den = cls ** dims
    if shape=='equal':
        return (n*den)**(1/dims)
    else:
        if dims==len(shape):
            raise ValueError('cannot define all the dimensions')
        if len(shape)==1:
            return ((n*den)/(shape[0]+cls-1))**(1/(dims-1))-cls+1
        else:
            return ((n*den)/((shape[0]+cls-1) * (shape[1]+cls-1)))-cls+1


def test_cls_set(err_exp, pf, clss, img_dims):
    err_fit = []
    for cls in clss:
        ns = ns_from_dims(img_dims, cls)
        err_model = bernouli(pf, ns)
        difference = abs(err_exp - err_model)
        err = np.mean(difference)
        err_fit.append(err)
    cls = clss[np.argmin(err_fit)].item()
    return cls


def tpc_fit(x, a, b, c):
    return a * np.e ** (-b * x) + c


def percentage_error(y_true, y_pred):  
    return (y_true - y_pred) / y_true


def mape(y_true, y_pred):  # mean absolute percentage error
    return np.mean(np.abs(percentage_error(y_true, y_pred))) 


def mape_linear_objective(params, y_pred, y_true):
    y_pred_new = linear_fit(y_pred, *params)
    return mape(y_true, y_pred_new) 


def linear_fit(x, m, b):
    return m * x + b 

def find_end_dist_tpc(pf, tpc, dist_arr):
    # print(f'pf^2 = {pf**2}')
    distances = np.concatenate([np.arange(0, np.max(dist_arr), 100)])
    # check the tpc change and the comparison to pf^2
    # over bigger and bigger discs:
    return find_end_dist_idx(pf, tpc, dist_arr, distances)
    

def find_end_dist_idx(pf, tpc, dist_arr, distances):
    """Finds the distance before the tpc function plateaus."""
    percentage = 0.05
    small_change = (pf-pf**2)*percentage 
    for dist_i in np.arange(1, len(distances)-1):
        start_dist, end_dist = distances[dist_i], distances[dist_i+1] 
        bool_array = (dist_arr>=start_dist) & (dist_arr<end_dist)
        sum_dev = np.sum(tpc[bool_array] - pf**2 > small_change)
        deviation = sum_dev/np.sum(bool_array)
        if deviation < 0.05:
            return distances[dist_i]
    return distances[1]


def tpc_to_cls(tpc, im, im_shape):
    '''Calculates the integral range from the tpc function.'''
    tpc = np.array(tpc)
    middle_idx = np.array(tpc.shape)//2
    pf = tpc[tuple(map(slice, middle_idx, middle_idx+1))].item()
    dist_arr_before = np.indices(tpc.shape)
    dist_arr_before = np.abs((dist_arr_before.T - middle_idx.T).T)
    img_volume = np.prod(im_shape)
    # normalising the tpc s.t. different vectors would have different weights,
    # According to their volumes.
    norm_vol = (np.array(im_shape).T - dist_arr_before.T).T
    norm_vol = np.prod(norm_vol, axis=0)/img_volume
    dist_arr = np.sqrt(np.sum(dist_arr_before**2, axis=0))
    end_dist = find_end_dist_tpc(pf, tpc, dist_arr)
    print(f'end dist = {end_dist}')
    pf_squared_end = np.mean(tpc[(dist_arr>=end_dist-10) & (dist_arr<=end_dist)])
    
    pf_squared = (pf_squared_end + pf**2)/2  
    bool_array = dist_arr<end_dist 
    
    # calculate the coefficient for the cls prediction:
    coeff = calc_coeff_for_cls_prediction(norm_vol, dist_arr, end_dist, img_volume, bool_array)
    pred_cls = calc_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape)
    pred_is_off, sign = pred_cls_is_off(pred_cls, im, pf)
    while pred_is_off:
        how_off = 'negative' if sign > 0 else 'positive'
        print(f'pred cls = {pred_cls} is too {how_off}, CHANGING TPC VALUES')
        tpc, pred_cls = change_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape, sign)
        pred_is_off, sign = pred_cls_is_off(pred_cls, im, pf)
    return pred_cls


def calc_coeff_for_cls_prediction(norm_vol, dist_arr, end_dist, img_volume, bool_array):
    sum_of_small_radii = np.sum(norm_vol[dist_arr<end_dist])
    coeff_1 = img_volume/(img_volume - sum_of_small_radii)
    coeff_2 = (1/img_volume)*(np.sum(bool_array)-np.sum(norm_vol[bool_array]))
    coeff_product = coeff_1*coeff_2
    while coeff_product > 1:
        print(f'coeff product = {coeff_product}')
        coeff_product /= 1.1
    return coeff_1/(1-coeff_product)


def change_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape, sign):
    '''Changes the tpc function to be more positive or more negative, compared
    to the fast stat. analysis cls pred. of the single img.'''
    if sign > 0:
        negatives = np.where(tpc - pf_squared < 0)
        tpc[negatives] += (pf_squared - tpc[negatives])/10
    else:
        positives = np.where(tpc - pf_squared > 0)
        tpc[positives] -= (tpc[positives] - pf_squared)/10
    pred_cls = calc_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape)
    return tpc, pred_cls


def calc_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape):
    pred_cls = coeff/(pf-pf_squared)*np.sum(tpc[bool_array] - pf_squared)
    if pred_cls > 0:
        pred_cls = pred_cls**(1/3) if len(im_shape)==3 else pred_cls**(1/2)
    return pred_cls


def pred_cls_is_off(pred_cls, img, pf):
    if pred_cls < 1:
        return True, 1
    one_im_stat_pred = one_img_stat_analysis_error(img, pf)
    if one_im_stat_pred > 1:  # could be erroneous stat. analysis prediction
        if pred_cls / one_im_stat_pred < 2/3:
            return True, 1
        if pred_cls / one_im_stat_pred > 2:
            return True, -1
    return False, 0


def fit_to_errs_function(dim, n_voxels, a, b):

    return a / n_voxels**b 


def make_error_prediction(img, conf=0.95, err_targ=0.05, model_error=True, mxtpc=100, shape='equal'):
    pf = torch.mean(img).item()
    dims = len(img.shape)
    # print(f'starting tpc radial')
    tpc = tpc_radial(img, threed=dims == 3, mx=mxtpc)
    cls = tpc_to_cls(tpc, img, img.shape)
    n = ns_from_dims([np.array(img.shape)], cls)
    # print(n, cls)
    std_bern = ((1 / n[0]) * (pf * (1 - pf))) ** 0.5
    std_model = get_std_model(dims, torch.numel(img))
    abs_err_target = err_targ * pf
    if model_error:
        # calculate the absolute error for the image:
        conf_bounds = get_prediction_interval(pf, std_bern, std_model, conf)
        abs_err_for_img = pf - conf_bounds[0]
        args = (conf, std_model, pf, abs_err_target)
        n_for_err_targ = minimize(optimize_error_n_pred, conf**0.5, args, bounds=bounds).fun
    else:  # TODO what is this useful for.. for when you trust the model completely?
        z = stats.norm.interval(conf)[1]
        abs_err_for_img = (z*std_bern/pf)
        n_for_err_targ = pf * (1 - pf) * (z/ (abs_err_target * pf)) ** 2

    l_for_err_targ = dims_from_n(n_for_err_targ, shape, cls, dims)
    percentage_err_for_img = abs_err_for_img/pf
    return percentage_err_for_img, l_for_err_targ, cls

def get_prediction_interval(image_pf, pred_std, pred_std_error_std, conf_level=0.95, n_divisions=101):
    '''Get the prediction interval for the phase fraction of the material given the image phase
    fraction, the predicted standard deviation and the standard deviation of the prediction error.'''
    # have a large enough number of stds to converge to 0 at both ends, 
    # but not too large to make the calculation slow:
    std_dist_std = pred_std*pred_std_error_std  # TODO see if this fits
    num_stds = min(pred_std/std_dist_std - pred_std/std_dist_std/10, 6)
    # First, make the "weights" or "error" distribution, the normal distribution of the stds
    # where the prediction std is the mean of this distribution:
    x_std_dist_bounds = [pred_std - num_stds*std_dist_std, pred_std + num_stds*std_dist_std]
    x_std_dist = np.linspace(*x_std_dist_bounds, n_divisions)
    std_dist = normal_dist(x_std_dist, mean=pred_std, std=std_dist_std)
    # Next, make the pf distributions, each row correspond to a different std, with 
    # the same mean (observed pf) but different stds (x_std_dist), multiplied by the
    # weights distribution (std_dist).
    pf_locs = np.ones((n_divisions,n_divisions))*image_pf
    pf_x_bounds = [image_pf - num_stds*pred_std, image_pf + num_stds*pred_std]
    pf_x_1d = np.linspace(*pf_x_bounds, n_divisions)
    pf_mesh, std_mesh = np.meshgrid(pf_x_1d, x_std_dist)
    # Before normalising by weight:
    pf_dist_before_norm = normal_dist(pf_mesh, mean=pf_locs, std=std_mesh)
    # Normalise by weight:
    pf_dist = (pf_dist_before_norm.T * std_dist).T    
    # Sum the distributions over the different stds
    sum_dist_norm = np.sum(pf_dist, axis=0)*np.diff(x_std_dist)[0]
    # Find the alpha confidence bounds
    cum_sum_sum_dist_norm = np.cumsum(sum_dist_norm*np.diff(pf_x_1d)[0])
    half_conf_level = (1+conf_level)/2
    conf_level_beginning = np.where(cum_sum_sum_dist_norm > 1-half_conf_level)[0][0]
    conf_level_end = np.where(cum_sum_sum_dist_norm > half_conf_level)[0][0]
    # Calculate the interval
    return pf_x_1d[conf_level_beginning], pf_x_1d[conf_level_end]

def normal_dist(x, mean, std):
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def optimize_error_conf_pred(bern_conf, total_conf, std_bern, std_model, pf):
    model_conf = total_conf/bern_conf
    err_bern = stats.norm.interval(bern_conf, scale=std_bern)[1]
    one_side_error_model = model_conf*2 - 1
    err_model = stats.norm.interval(one_side_error_model, scale=std_model)[1]
    return err_bern * (1 + err_model)


def optimize_error_n_pred(bern_conf, total_conf, std_model, pf, err_targ):
    model_conf = total_conf/bern_conf
    z1 = stats.norm.interval(bern_conf)[1]
    one_side_error_model = model_conf*2 - 1
    err_model = stats.norm.interval(one_side_error_model, scale=std_model)[1]
    num = (err_model+1)**2 * (1-pf) * z1**2 * pf
    den = (err_targ)**2  # TODO go over the calcs and see if this is right
    return num/den


def get_std_model(dim, n_voxels): 
    popt = {'2d': [48.20175315, 0.4297919],
            '3d': [444.803518, 0.436974444]} 
    return fit_to_errs_function(dim, n_voxels, *popt[f'{dim}d'])


def calc_autocorrelation_orthant(img, numel_large, dims, desired_length=100):
    """Calculates the autocorrelation function of an image using the FFT method
    Calculates over a single orthant, in all directions right and down from the origin."""
    ax = list(range(0, len(img.shape)))
    img_FFT = np.fft.rfftn(img, axes=ax)
    tpc = np.fft.irfftn(img_FFT.conjugate() * img_FFT, s=img.shape, axes=ax).real / numel_large
    return tpc[tuple(map(slice, (desired_length,)*dims))]


def two_point_correlation_orthant(img, dims, desired_length=100, periodic=True):
    """
    Calculates the two point correlation function of an image along an orthant.
    If the image is not periodic, it pads the image with desired_length number of zeros, before
    before calculating the 2PC function using the FFT method. After the FFT calculation, it 
    normalises the result by the number of possible occurences of the 2PC function.
    """
    if not periodic:  # padding the image with zeros, then calculates the normaliser.
        indices_img = np.indices(img.shape) + 1
        normaliser = np.flip(np.prod(indices_img, axis=0))
        img = np.pad(img, [(0, desired_length) for _ in range(dims)], 'constant')
    numel_large = np.product(img.shape)
    tpc_desired = calc_autocorrelation_orthant(img, numel_large, dims, desired_length)
    if not periodic:  # normalising the result
        normaliser = normaliser[tuple(map(slice, tpc_desired.shape))]
        normaliser = numel_large/normaliser
        return normaliser*tpc_desired
    else:
        return tpc_desired  
    

def two_point_correlation(img, desired_length=100, periodic=True, threed=False):
    """ 
    Calculates the two point correlation function of an image using the FFT method. 
    The crosscorrelation function is calculated in all directions, where the middle of the output
    is the origin f(0,0,0) (or f(0,0) in 2D). 
    :param img: the image to calculate the 2PC function of.
    :param desired_length: the length of the 2PC function to calculate. Preferably even.
    :param periodic: whether the image is periodic or not, and whether the FFT calculation is done
    with the periodic assumption.
    :param threed: whether the image is 3D or not.
    """
    img = np.array(img)
    dims = 3 if threed else 2
    orthants = {}
    # calculating the 2PC function for each orthant, and saving the result in a dictionary:
    for axis in product((1, 0), repeat=dims-1):
        flip_list = np.arange(dims-1)[~np.array(axis, dtype=bool)]
        # flipping the orthant to the opposite side for calculation of the 2PC:
        flipped_img = np.flip(img, flip_list)
        tpc_orthant = two_point_correlation_orthant(flipped_img, dims, desired_length+1, periodic)
        backflip_orthant = np.flip(tpc_orthant, flip_list)
        orthants[axis + (1,)] = backflip_orthant
        opposite_axis = tuple(1 - np.array(axis)) + (0,)
        orthants[opposite_axis] = np.flip(backflip_orthant)  # flipping the orthant to the opposite side
    res = np.zeros((desired_length*2+1,)*dims)
    for axis in orthants.keys():
        axis_idx = np.array(axis)*desired_length
        slice_to_input = tuple(map(slice, axis_idx, axis_idx+desired_length+1))
        res[slice_to_input] = orthants[axis]
    return res


def one_img_stat_analysis_error(img, pf): 
    return stat_analysis_error_classic(img.unsqueeze(0), pf)
    

def calc_std_from_ratio(img, ratio):
    """Calculates the standard deviation of the subimages of an image, divided by a certain ratio."""
    divided_img = divide_img_to_subimages(img, ratio).cpu().numpy()
    along_axis = tuple(np.arange(1, len(img.shape)))
    ddof = 1  # for unbiased std
    return np.std(np.mean(divided_img, axis=along_axis), ddof=ddof)


def image_stats(img, pf, ratios, z_score=1.96):  
    errs = []
    for ratio in ratios:
        std_ratio = calc_std_from_ratio(img, ratio)
        errs.append(100 * ((z_score * std_ratio) / pf))
    return errs
    

def divide_img_to_subimages(img, subimg_ratio):
    """Divides an image to subimages from a certain ratio."""
    threed = len(img.shape) == 4
    one_img_shape = np.array(img.shape)[1:]
    subimg_shape = one_img_shape // subimg_ratio
    n_images = one_img_shape // subimg_shape
    im_to_divide_size = n_images * subimg_shape
    im_to_divide_size = np.insert(im_to_divide_size, 0, img.shape[0])
    im_to_divide = img[tuple(map(slice, im_to_divide_size))]
    reshape_shape = list(chain.from_iterable(zip(n_images, subimg_shape)))
    im_to_divide = im_to_divide.reshape(img.shape[0],*reshape_shape)
    im_to_divide = im_to_divide.swapaxes(2,3)
    if threed:
        im_to_divide = im_to_divide.swapaxes(4,5).swapaxes(3,4)
    return im_to_divide.reshape((np.prod(n_images)*img.shape[0], *subimg_shape))
    

if __name__ == '__main__':
    img = np.arange(4*5*6).reshape(4,5,6)
    div_im = divide_img_to_subimages(img, 2)
    print('hi')

# def cls(img):
#     """
#     Calculating the chord length distribution function
#     """
#     iterations = 150
#     return_length = 150
#     sums = torch.zeros(iterations)

#     # for ang in torch.linspace(0,180, 20):
#     sm = []
#     # cur_im = rotate(torch.tensor(img), ang.item())
#     # cur_im = torch.round(cur_im[0,0,280:-280, 280:-280])
#     cur_im = torch.tensor(img, device=torch.device("cuda:0"))
#     for i in range(1, iterations):
#         sm.append(
#             torch.sum(cur_im)
#         )  # sum of all current "live" pixels that are part of an i length chord
#         cur_im = (
#             cur_im[1:] * cur_im[:-1]
#         )  # deleting 1 pixel for each chord, leaving all pixels that are part of an i+1 length chords
#     sm.append(torch.sum(cur_im))
#     sums += torch.tensor(sm)

#     cls = sums.clone()
#     cls[-1] = 0  # the assumption is that the last sum is 0
#     for i in range(1, iterations):  # calculation of the chord lengths by the sums
#         cls[-(i + 1)] = (sums[-(i + 1)] - sums[-i] - sum(cls[-i:])).cpu().item()
#     cls = np.array(cls)
#     return cls / np.sum(cls)


# def make_error_prediction_old(img, conf=0.95, err_targ=0.05,  model_error=True, correction=True, mxtpc=100, shape='equal', met='pf'):
#     pf = img.mean()
#     dims = len(img.shape)
#     tpc = tpc_radial(img, threed=dims == 3, mx=mxtpc)
#     cut = max(20, np.argmin(tpc))
#     tpc = tpc[:cut]
#     x = np.arange(len(tpc))
#     cls = tpc_to_cls(x, tpc)
#     n = ns_from_dims([np.array(img.shape)], cls)
#     # print(n, cls)
#     std_bern = ((1 / n[0]) * (pf * (1 - pf))) ** 0.5
#     std_model, slope, intercept = get_model_params(f'{dims}d{met}') 
#     if not correction:
#         slope, intercept = 1, 0
#     if model_error:
#         # print(std_bern)
#         bounds = [(conf*1.001, 1)]
#         args = (conf, std_bern, std_model, pf, slope, intercept)
#         err_for_img = minimize(optimize_error_conf_pred, conf**0.5, args, bounds=bounds).fun
#         args = (conf, std_model, pf, slope, intercept, err_targ)
#         n_for_err_targ = minimize(optimize_error_n_pred, conf**0.5, args, bounds=bounds).fun
#         # print(n, n_for_err_targ, cls)
#     else:
#         z = stats.norm.interval(conf)[1]
#         err_for_img = (z*std_bern/pf)*slope+intercept
#         # print(stats.norm.interval(conf, scale=std_bern)[1], std_bern)
#         n_for_err_targ = pf * (1 - pf) * (z/ ((err_targ -intercept)/slope * pf)) ** 2

#         # print(n_for_err_targ, n, cls)
#     l_for_err_targ = dims_from_n(n_for_err_targ, shape, cls, dims)
#     return err_for_img, l_for_err_targ, tpc

#  Calc tpc radial the old way without fft:

        # else:
        #     img = torch.tensor(img, device=torch.device("cuda:0")).float()
        # tpc = {i:[0,0] for i in range(mx+1)}
        # for x in range(0, mx):
        #     for y in range(0, mx):
        #         for z in range(0, mx if threed else 1):
        #             d = (x**2 + y**2 + z**2) ** 0.5
        #             if d < mx:
        #                 remainder = d%1
        #                 if w_fft:
        #                     cur_tpc = tpc_radial[x,y,z] if threed else tpc_radial[x,y]
        #                 else:
        #                     con_img = conjunction_img_for_tpc(img, x, y, z, threed)
        #                     cur_tpc = torch.mean(con_img).cpu()
        #                 weight_floor = 1-remainder
        #                 weight_ceil = remainder
        #                 tpc[int(d)][0] += weight_floor 
        #                 tpc[int(d)][1] += cur_tpc*weight_floor
        #                 tpc[int(d)+1][0] += weight_ceil 
        #                 tpc[int(d)+1][1] += cur_tpc*weight_ceil
    
        # tpcfin = [tpc[key][1]/tpc[key][0] for key in tpc.keys()]
        # tpcfin = np.array(tpcfin, dtype=np.float64)
        # tpcfin_list.append(tpcfin)
    

    
        # else:
        #     img = torch.tensor(img, device=torch.device("cuda:0")).float()
        # tpc = {i:[0,0] for i in range(mx+1)}
        # for x in range(0, mx):
        #     for y in range(0, mx):
        #         for z in range(0, mx if threed else 1):
        #             d = (x**2 + y**2 + z**2) ** 0.5
        #             if d < mx:
        #                 remainder = d%1
        #                 if w_fft:
        #                     cur_tpc = tpc_radial[x,y,z] if threed else tpc_radial[x,y]
        #                 else:
        #                     con_img = conjunction_img_for_tpc(img, x, y, z, threed)
        #                     cur_tpc = torch.mean(con_img).cpu()
        #                 weight_floor = 1-remainder
        #                 weight_ceil = remainder
        #                 tpc[int(d)][0] += weight_floor 
        #                 tpc[int(d)][1] += cur_tpc*weight_floor
        #                 tpc[int(d)+1][0] += weight_ceil 
        #                 tpc[int(d)+1][1] += cur_tpc*weight_ceil
    
        # tpcfin = [tpc[key][1]/tpc[key][0] for key in tpc.keys()]
        # tpcfin = np.array(tpcfin, dtype=np.float64)
        # tpcfin_list.append(tpcfin)


# def make_sa_old(img, batch=True):
#     if not batch:
#         img = np.expand_dims(img, 0)
#         sa = np.zeros_like(img)
#     else:
#         sa = torch.zeros_like(img)
#     dims = len(img.shape)
#     sa[:, 1:] += img[:, 1:] * (1 - img[:, :-1])
#     sa[:, :, 1:] += img[:, :, 1:] * (1 - img[:, :, :-1])
#     sa[:, :-1] += (1 - img[:, 1:]) * img[:, :-1]
#     sa[:, :, :-1] += (1 - img[:, :, 1:]) * img[:, :, :-1]
#     if dims == 4:
#         sa[:, :, :, 1:] += img[:, :, :, 1:] * (1 - img[:, :, :, :-1])
#         sa[:, :, :, :-1] += (1 - img[:, :, :, 1:]) * img[:, :, :, :-1]
#     if not batch:
#         sa = sa[0]
#     return sa


# def tpc_radial_fft(img_list, mx=100, threed=False):
#     """Calculates the radial tpc using fft"""
#     tpcfin_list = []
#     for i in range(len(img_list)):  
#         img = img_list[i]
#         tpc_radial = p2_crosscorrelation(img, img)
#         if threed:
#             tpc_radial = tpc_radial[:mx, :mx, :mx]
#         else:   
#             tpc_radial = tpc_radial[:mx, :mx]
#         tpcfin_list.append(np.array(tpc_radial, dtype=np.float64))
#     return tpcfin_list


# def make_sas(img, batch=True):  # TODO check this works
#     if not batch:
#         img = torch.unsqueeze(img, 0)
#     sa_lr = torch.zeros_like(img)
#     sa_ud = torch.zeros_like(img)
#     dims = len(img.shape)
#     sa_lr[:, :, :-1] = (img[:, :, 1:] + img[:, :, :-1]) % 2
#     sa_ud[:, :-1] = (img[:, 1:] + img[:, :-1]) % 2
#     sas = [sa_lr, sa_ud]
#     if dims == 4:  # 3d
#         sa_z = torch.zeros_like(img)
#         sa_z[:, :, :, :-1] += (img[:, :, :, 1:] + img[:, :, :, :-1]) % 2
#         sas.append(sa_z)
#     if not batch:
#         sas = [sa[0] for sa in sas]
#     return sas


# def sa_map_from_sas(sa_images):
#     # For memory issues, this calculation is the same as:
#     # return torch.stack(sa_images, dim=0).sum(dim=0)
#     sa_maps = [[sa_images[map_idx][k, ...] for map_idx in range(len(sa_images))] for k in range(sa_images[0].size()[0])]
#     return torch.stack([torch.stack(maps, dim=0).sum(dim=0) for maps in sa_maps], dim=0)


# def make_sa(img, batch=True):
#     if not batch:
#         img = np.expand_dims(img, 0)
#         sa = np.zeros_like(img)
#     else:
#         sa = torch.zeros_like(img)
#     dims = len(img.shape)
#     sa[:, 1:] += img[:, 1:] * (1 - img[:, :-1])
#     sa[:, :, 1:] += img[:, :, 1:] * (1 - img[:, :, :-1])
#     sa[:, :-1] += (1 - img[:, 1:]) * img[:, :-1]
#     sa[:, :, :-1] += (1 - img[:, :, 1:]) * img[:, :, :-1]
#     if dims == 4:
#         sa[:, :, :, 1:] += img[:, :, :, 1:] * (1 - img[:, :, :, :-1])
#         sa[:, :, :, :-1] += (1 - img[:, :, :, 1:]) * img[:, :, :, :-1]
#     if not batch:
#         sa = sa[0]
#     sa[sa>0] = 1
#     return sa