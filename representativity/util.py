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


def tpc_radial(img, mx=100, threed=False):
    desired_length = img.shape[0]//2
    return two_point_correlation(img, desired_length=desired_length, periodic=True, threed=threed)


def stat_analysis_error_classic(img, vf):  # TODO see if to delete this or not
    ratios = [2**i for i in np.arange(1, int(np.log2(img.shape[1]))-5)]
    ratios.reverse()
    if img.shape[0] > 1:
        ratios.append(1)
    ratios = ratios[-4:]
    edge_lengths = [img.shape[1]//r for r in ratios]
    img_dims = [np.array((l,)*(len(img.shape)-1)) for l in edge_lengths]
    err_exp = image_stats(img, vf, ratios)
    real_ir = fit_ir(err_exp, img_dims, vf)
    # TODO different size image 1000 vs 1500
    return real_ir


def stat_analysis_error(img, vf, edge_lengths):  # TODO see if to delete this or not
    img_dims = [np.array((l,)*(len(img.shape)-1)) for l in edge_lengths]
    err_exp = real_image_stats(img, edge_lengths, vf)
    real_ir = fit_ir(err_exp, img_dims, vf)
    # TODO different size image 1000 vs 1500
    return real_ir


def real_image_stats(img, ls, vf, repeats=4000, z_score=1.96):  
    dims = len(img[0].shape)
    errs = []
    for l in ls:
        vfs = []
        n_pos_ims = int(np.prod(img.shape)/l**dims)
        repeats = n_pos_ims*2
        print(f'one im repeats = {repeats} for l = {l}')
        if dims == 1:
            for _ in range(repeats):
                bm, xm = img.shape
                x = torch.randint(0, xm - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l]
                vfs.append(torch.mean(crop).cpu())
        elif dims == 2:
            for _ in range(repeats):
                bm, xm, ym = img.shape
                x = torch.randint(0, xm - l, (1,))
                y = torch.randint(0, ym - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l, y : y + l]
                vfs.append(torch.mean(crop).cpu())
        else:  # 3D
            for _ in range(repeats):
                bm, xm, ym, zm = img.shape
                x = torch.randint(0, xm - l, (1,))
                y = torch.randint(0, ym - l, (1,))
                z = torch.randint(0, zm - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l, y : y + l, z : z + l]
                vfs.append(torch.mean(crop).cpu())
        vfs = np.array(vfs)
        ddof = np.ceil(repeats/img.shape[0])
        print(f'ddof = {ddof}')
        std = np.std(vfs, ddof=ddof)
        errs.append(100 * ((z_score * std) / vf))
    return errs


def bernouli(vf, ns, conf=0.95):
    errs = []
    for n in ns:
        std_theo = ((1 / n) * (vf * (1 - vf))) ** 0.5
        errs.append(100 * (stats.norm.interval(conf, scale=std_theo)[1] / vf))
    return np.array(errs, dtype=np.float64)


def fit_ir(err_exp, img_dims, vf, max_ir=150):
    err_exp = np.array(err_exp)
    ir = test_ir_set(err_exp, vf, np.arange(1, max_ir, 1), img_dims)
    ir = test_ir_set(err_exp, vf, np.linspace(ir - 1, ir + 1, 50), img_dims)
    # print(f'real ir = {ir}')
    return ir


def ns_from_dims(img_dims, ir):
    n_dims = len(img_dims[0])
    den = ir ** n_dims
    # return [np.prod(np.array(i)) / den for i in img_dims]
    return [np.prod(np.array(i)) / den for i in img_dims]
    # if n_dims == 3:  # 2ir length
    #     return [np.prod(i + 2*(ir - 1)) / den for i in img_dims]
    # else:  # n_dims == 2
    #     return [np.prod(i + ir - 1) / den for i in img_dims]

def dims_from_n(n, shape, ir, dims):
    den = ir ** dims
    if shape=='equal':
        return (n*den)**(1/dims)-ir+1
    else:
        if dims==len(shape):
            raise ValueError('cannot define all the dimensions')
        if len(shape)==1:
            return ((n*den)/(shape[0]+ir-1))**(1/(dims-1))-ir+1
        else:
            return ((n*den)/((shape[0]+ir-1) * (shape[1]+ir-1)))-ir+1


def test_ir_set(err_exp, vf, irs, img_dims):
    err_fit = []
    for ir in irs:
        ns = ns_from_dims(img_dims, ir)
        err_model = bernouli(vf, ns)
        err = np.mean(abs(err_exp - err_model))
        err_fit.append(err)
    ir = irs[np.argmin(err_fit)].item()
    return ir


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

def find_end_dist_tpc(vf, tpc, dist_arr):
    # print(f'vf^2 = {vf**2}')
    distances = np.concatenate([np.arange(0, np.max(dist_arr), 100)])
    # check the tpc change and the comparison to vf^2
    # over bigger and bigger discs:
    return find_end_dist_idx(vf, tpc, dist_arr, distances)
    

def find_end_dist_idx(vf, tpc, dist_arr, distances):
    """Finds the distance before the tpc function plateaus."""
    percentage = 0.05
    small_change = (vf-vf**2)*percentage 
    for dist_i in np.arange(1, len(distances)-1):
        start_dist, end_dist = distances[dist_i], distances[dist_i+1] 
        bool_array = (dist_arr>=start_dist) & (dist_arr<end_dist)
        sum_dev = np.sum(tpc[bool_array] - vf**2 > small_change)
        deviation = sum_dev/np.sum(bool_array)
        if deviation < 0.05:
            return distances[dist_i]
    return distances[1]


def tpc_to_ir(tpc, im, im_shape):
    '''Calculates the integral range from the tpc function.'''
    tpc = np.array(tpc)
    middle_idx = np.array(tpc.shape)//2
    vf = tpc[tuple(map(slice, middle_idx, middle_idx+1))].item()
    # print(f'vf squared = {np.round(vf**2, 5)}')
    dist_arr_before = np.indices(tpc.shape)
    dist_arr_before = np.abs((dist_arr_before.T - middle_idx.T).T)
    img_volume = np.prod(im_shape)
    # normalising the tpc s.t. different vectors would have different weights,
    # According to their volumes.
    norm_vol = (np.array(im_shape).T - dist_arr_before.T).T
    norm_vol = np.prod(norm_vol, axis=0)/img_volume
    dist_arr = np.sqrt(np.sum(dist_arr_before**2, axis=0))
    end_dist = find_end_dist_tpc(vf, tpc, dist_arr)
    print(f'end dist = {end_dist}')
    vf_squared_end = np.mean(tpc[(dist_arr>=end_dist-10) & (dist_arr<=end_dist)])
    
    # print(f'end of tpc = {np.round(vf_squared_end, 5)}')
    vf_squared = (vf_squared_end + vf**2)/2  
    bool_array = dist_arr<end_dist 
    
    # calculate the coefficient for the ir prediction:
    coeff = calc_coeff_for_ir_prediction(norm_vol, dist_arr, end_dist, img_volume, bool_array)
    # print(f'ir pred coefficient = {coeff}')
    pred_ir = calc_pred_ir(coeff, tpc, vf, vf_squared, bool_array, im_shape)
    pred_is_off, sign = pred_ir_is_off(pred_ir, im, vf)
    while pred_is_off:
        how_off = 'negative' if sign > 0 else 'positive'
        print(f'pred ir = {pred_ir} is too {how_off}, CHANGING TPC VALUES')
        tpc, pred_ir = change_pred_ir(coeff, tpc, vf, vf_squared, bool_array, im_shape, sign)
        pred_is_off, sign = pred_ir_is_off(pred_ir, im, vf)
    return pred_ir


def calc_coeff_for_ir_prediction(norm_vol, dist_arr, end_dist, img_volume, bool_array):
    sum_of_small_radii = np.sum(norm_vol[dist_arr<end_dist])
    coeff_1 = img_volume/(img_volume - sum_of_small_radii)
    coeff_2 = (1/img_volume)*(np.sum(bool_array)-np.sum(norm_vol[bool_array]))
    coeff_product = coeff_1*coeff_2
    while coeff_product > 1:
        print(f'coeff product = {coeff_product}')
        coeff_product /= 1.1
    return coeff_1/(1-coeff_product)


def change_pred_ir(coeff, tpc, vf, vf_squared, bool_array, im_shape, sign):
    '''Changes the tpc function to be more positive or more negative, compared
    to the fast stat. analysis ir pred. of the single img.'''
    if sign > 0:
        negatives = np.where(tpc - vf_squared < 0)
        tpc[negatives] += (vf_squared - tpc[negatives])/10
    else:
        positives = np.where(tpc - vf_squared > 0)
        tpc[positives] -= (tpc[positives] - vf_squared)/10
    pred_ir = calc_pred_ir(coeff, tpc, vf, vf_squared, bool_array, im_shape)
    return tpc, pred_ir


def calc_pred_ir(coeff, tpc, vf, vf_squared, bool_array, im_shape):
    pred_ir = coeff/(vf-vf_squared)*np.sum(tpc[bool_array] - vf_squared)
    if pred_ir > 0:
        pred_ir = pred_ir**(1/3) if len(im_shape)==3 else pred_ir**(1/2)
    return pred_ir


def pred_ir_is_off(pred_ir, img, vf):
    if pred_ir < 1:
        return True, 1
    one_im_stat_pred = one_img_stat_analysis_error(img, vf)
    if one_im_stat_pred > 1:  # could be erroneous stat. analysis prediction
        if pred_ir / one_im_stat_pred < 2/3:
            return True, 1
        if pred_ir / one_im_stat_pred > 2:
            return True, -1
    return False, 0


def fit_to_errs_function(dim, n_voxels, a, b):

    return a / n_voxels**b 


def make_error_prediction(img, conf=0.95, err_targ=0.05, model_error=True, correction=True, mxtpc=100, shape='equal', met='vf'):
    vf = torch.mean(img).item()
    dims = len(img.shape)
    # print(f'starting tpc radial')
    tpc = tpc_radial(img, threed=dims == 3, mx=mxtpc)
    ir = tpc_to_ir(tpc, img, img.shape)
    n = ns_from_dims([np.array(img.shape)], ir)
    # print(n, ir)
    std_bern = ((1 / n[0]) * (vf * (1 - vf))) ** 0.5
    std_model, slope = get_model_params(dims, torch.numel(img))
    if not correction:
        slope = 1 
    if model_error:
        # print(std_bern)
        bounds = [(conf*1.001, 1)]
        args = (conf, std_bern, std_model, vf, slope)
        err_for_img = minimize(optimize_error_conf_pred, conf**0.5, args, bounds=bounds).fun
        args = (conf, std_model, vf, slope, err_targ)
        n_for_err_targ = minimize(optimize_error_n_pred, conf**0.5, args, bounds=bounds).fun
        # print(n, n_for_err_targ, ir)
    else:
        z = stats.norm.interval(conf)[1]
        err_for_img = (z*std_bern/vf)*slope
        # print(stats.norm.interval(conf, scale=std_bern)[1], std_bern)
        n_for_err_targ = vf * (1 - vf) * (z/ (err_targ/slope * vf)) ** 2

        # print(n_for_err_targ, n, ir)
    l_for_err_targ = dims_from_n(n_for_err_targ, shape, ir, dims)
    return err_for_img*100, l_for_err_targ, ir


def optimize_error_conf_pred(bern_conf, total_conf, std_bern, std_model, vf, slope):
    model_conf = total_conf/bern_conf
    err_bern = ((stats.norm.interval(bern_conf, scale=std_bern)[1]*slope)/vf)
    err_model = stats.norm.interval(model_conf, scale=std_model)[1]
    # print(stats.norm.interval(bern_conf, scale=std_bern)[1], slope, vf , err_model, err_bern)
    # print(err_bern, err_model, err_bern * (1 + err_model))
    # print(err_bern * (1 + err_model))
    return err_bern * (1 + err_model)


def optimize_error_n_pred(bern_conf, total_conf, std_model, vf, slope, err_targ):
    # print(bern_conf)
    model_conf = total_conf/bern_conf
    z1 = stats.norm.interval(bern_conf)[1]
    err_model = stats.norm.interval(model_conf, scale=std_model)[1]
    # print(err_model)
    num = -(err_model+1)**2 * slope**2 * (vf-1) * z1**2
    den = (err_model + err_targ)**2 * vf
    return num/den


def get_model_params(dim, n_voxels):  # see model_param.py for the appropriate code that was used.
    params= {f'{dim}d': [fit_to_errs_function(dim, n_voxels, 48.20175315, 0.4297919), 1],
             f'{dim}d': [0.5584825884176943, 6]}  # TODO needs to be changed according to the fit in prediction_error.py
    return params[f'{dim}d']


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


def one_img_stat_analysis_error(img, vf): 
    return stat_analysis_error_classic(img.unsqueeze(0), vf)
    

def calc_std_from_ratio(img, ratio):
    """Calculates the standard deviation of the subimages of an image, divided by a certain ratio."""
    divided_img = divide_img_to_subimages(img, ratio).cpu().numpy()
    along_axis = tuple(np.arange(1, len(img.shape)))
    ddof = np.prod(np.array(img.shape[1:])//np.array(divided_img.shape[1:]))
    return np.std(np.mean(divided_img, axis=along_axis), ddof=ddof)


def image_stats(img, vf, ratios, z_score=1.96):  
    errs = []
    for ratio in ratios:
        std_ratio = calc_std_from_ratio(img, ratio)
        errs.append(100 * ((z_score * std_ratio) / vf))
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

# def cld(img):
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

#     cld = sums.clone()
#     cld[-1] = 0  # the assumption is that the last sum is 0
#     for i in range(1, iterations):  # calculation of the chord lengths by the sums
#         cld[-(i + 1)] = (sums[-(i + 1)] - sums[-i] - sum(cld[-i:])).cpu().item()
#     cld = np.array(cld)
#     return cld / np.sum(cld)


# def make_error_prediction_old(img, conf=0.95, err_targ=0.05,  model_error=True, correction=True, mxtpc=100, shape='equal', met='vf'):
#     vf = img.mean()
#     dims = len(img.shape)
#     tpc = tpc_radial(img, threed=dims == 3, mx=mxtpc)
#     cut = max(20, np.argmin(tpc))
#     tpc = tpc[:cut]
#     x = np.arange(len(tpc))
#     ir = tpc_to_ir(x, tpc)
#     n = ns_from_dims([np.array(img.shape)], ir)
#     # print(n, ir)
#     std_bern = ((1 / n[0]) * (vf * (1 - vf))) ** 0.5
#     std_model, slope, intercept = get_model_params(f'{dims}d{met}') 
#     if not correction:
#         slope, intercept = 1, 0
#     if model_error:
#         # print(std_bern)
#         bounds = [(conf*1.001, 1)]
#         args = (conf, std_bern, std_model, vf, slope, intercept)
#         err_for_img = minimize(optimize_error_conf_pred, conf**0.5, args, bounds=bounds).fun
#         args = (conf, std_model, vf, slope, intercept, err_targ)
#         n_for_err_targ = minimize(optimize_error_n_pred, conf**0.5, args, bounds=bounds).fun
#         # print(n, n_for_err_targ, ir)
#     else:
#         z = stats.norm.interval(conf)[1]
#         err_for_img = (z*std_bern/vf)*slope+intercept
#         # print(stats.norm.interval(conf, scale=std_bern)[1], std_bern)
#         n_for_err_targ = vf * (1 - vf) * (z/ ((err_targ -intercept)/slope * vf)) ** 2

#         # print(n_for_err_targ, n, ir)
#     l_for_err_targ = dims_from_n(n_for_err_targ, shape, ir, dims)
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