import numpy as np
from itertools import product, chain
from scipy.stats import norm
from scipy.optimize import minimize


def autocorrelation_orthant(
    binary_img: np.ndarray, num_elements: int, n_dims: int, desired_length: int = 100
):
    """Calculates the autocorrelation function of an image using the FFT method
    Calculates over a single orthant, in all directions right and down from the origin.
    """
    ax = list(range(0, len(binary_img.shape)))
    img_FFT = np.fft.rfftn(binary_img, axes=ax)
    tpc = (
        np.fft.irfftn(img_FFT.conjugate() * img_FFT, s=binary_img.shape, axes=ax).real
        / num_elements
    )
    return tpc[tuple(map(slice, (desired_length,) * n_dims))]


def two_point_correlation_orthant(
    binary_img: np.ndarray,
    n_dims: int,
    desired_length: int = 100,
    periodic: bool = True,
):
    """
    Calculates the two point correlation function of an image along an orthant.
    If the image is not periodic, it pads the image with desired_length number of zeros, before
    before calculating the 2PC function using the FFT method. After the FFT calculation, it
    normalises the result by the number of possible occurences of the 2PC function.
    """
    img_shape = binary_img.shape
    if not periodic:  # padding the image with zeros, then calculates the normaliser.
        indices_img = np.indices(img_shape) + 1
        normaliser = np.flip(np.prod(indices_img, axis=0))
        binary_img = np.pad(
            binary_img, [(0, desired_length) for _ in range(n_dims)], "constant"
        )
    num_elements = np.product(img_shape)
    tpc_desired = autocorrelation_orthant(
        binary_img, num_elements, n_dims, desired_length
    )

    if not periodic:  # normalising the result
        normaliser = normaliser[tuple(map(slice, tpc_desired.shape))]
        normaliser = num_elements / normaliser
        return normaliser * tpc_desired
    else:
        return tpc_desired


def two_point_correlation(
    binary_img: np.ndarray,
    desired_length: int = 100,
    volumetric: bool = False,
    periodic: bool = True,
) -> np.ndarray:
    n_dims = 3 if volumetric else 2
    # orthant = N-D quadrant. stored here, indexed by axis
    orthants: dict[tuple, np.ndarray] = {}
    # calculating the 2PC function for each orthant, saving the result in a dictionary
    for axis in product((1, 0), repeat=n_dims - 1):
        flip_list = np.arange(n_dims - 1)[~np.array(axis, dtype=bool)]
        # flipping img to the opposite side for calculation of the 2PC:
        flipped_img = np.flip(binary_img, flip_list)

        tpc_orthant = two_point_correlation_orthant(
            flipped_img, n_dims, desired_length, periodic
        )
        original_tpc_orthant = np.flip(tpc_orthant, flip_list)
        orthants[axis + (1,)] = original_tpc_orthant

        # flipping the orthant to the opposite side
        opposite_axis = tuple(1 - np.array(axis)) + (0,)
        orthants[opposite_axis] = np.flip(original_tpc_orthant)

    result = np.zeros((desired_length * 2 + 1,) * n_dims)
    for axis in orthants.keys():
        axis_idx = np.array(axis) * desired_length
        slice_to_input = tuple(map(slice, axis_idx, axis_idx + desired_length))  # + 1
        result[slice_to_input] = orthants[axis]
    return result


def radial_tpc(
    binary_img: np.ndarray, volumetric: bool = False, periodic: bool = True
) -> np.ndarray:

    img_y_length: int = binary_img.shape[0]  # was 0
    # desired length: output of fft
    desired_length = (img_y_length // 2) if periodic else (img_y_length - 1)
    return two_point_correlation(
        binary_img,
        desired_length=desired_length,
        volumetric=volumetric,
        periodic=periodic,
    )


def calc_coeff_for_cls_prediction(
    norm_vol: np.ndarray,
    dist_arr: np.ndarray,
    end_dist: float,
    img_volume: int,
    bool_array: np.ndarray,
) -> float:
    sum_of_small_radii = np.sum(norm_vol[dist_arr < end_dist])
    coeff_1 = img_volume / (img_volume - sum_of_small_radii)
    coeff_2 = (1 / img_volume) * (np.sum(bool_array) - np.sum(norm_vol[bool_array]))
    coeff_product = coeff_1 * coeff_2
    while coeff_product > 1:
        print(f"coeff product = {coeff_product}")
        coeff_product /= 1.1
    return coeff_1 / (1 - coeff_product)


def find_end_dist_idx(
    pf: float, tpc: np.ndarray, dist_arr: np.ndarray, distances: np.ndarray
):
    """Finds the distance before the tpc function plateaus."""
    percentage = 0.05
    small_change = (pf - pf**2) * percentage
    for dist_i in np.arange(1, len(distances) - 1):
        start_dist, end_dist = distances[dist_i], distances[dist_i + 1]
        bool_array = (dist_arr >= start_dist) & (dist_arr < end_dist)
        sum_dev = np.sum(tpc[bool_array] - pf**2 > small_change)
        deviation = sum_dev / np.sum(bool_array)
        if deviation < 0.05:
            return distances[dist_i]
    return distances[1]


def find_end_dist_tpc(pf: float, tpc: np.ndarray, dist_arr: np.ndarray) -> float:
    # print(f'pf^2 = {pf**2}')
    distances = np.linspace(
        0, int(np.max(dist_arr)), 100
    )  # np.concatenate([np.arange(0, np.max(dist_arr), 100)])
    # check the tpc change and the comparison to pf^2
    # over bigger and bigger discs:
    print(f"distances: {distances.shape}")
    return find_end_dist_idx(pf, tpc, dist_arr, distances)


def calc_pred_cls(
    coeff: float,
    tpc: np.ndarray,
    pf: float,
    pf_squared: float,
    bool_array: np.ndarray,
    im_shape: tuple[int, ...],
) -> float:
    pred_cls = coeff / (pf - pf_squared) * np.sum(tpc[bool_array] - pf_squared)
    if pred_cls > 0:
        pred_cls = pred_cls ** (1 / 3) if len(im_shape) == 3 else pred_cls ** (1 / 2)
    return pred_cls


def divide_img_to_subimages(img: np.ndarray, subimg_ratio) -> np.ndarray:
    """Divides an image to subimages from a certain ratio."""
    img = img[np.newaxis, :]
    threed = len(img.shape) == 4
    one_img_shape = np.array(img.shape)[1:]
    subimg_shape = one_img_shape // subimg_ratio
    n_images = one_img_shape // subimg_shape
    im_to_divide_size = n_images * subimg_shape
    im_to_divide_size = np.insert(im_to_divide_size, 0, img.shape[0])
    im_to_divide = img[tuple(map(slice, im_to_divide_size))]
    reshape_shape = list(chain.from_iterable(zip(n_images, subimg_shape)))
    im_to_divide = im_to_divide.reshape(img.shape[0], *reshape_shape)
    im_to_divide = im_to_divide.swapaxes(2, 3)
    if threed:
        im_to_divide = im_to_divide.swapaxes(4, 5).swapaxes(3, 4)
    return im_to_divide.reshape((np.prod(n_images) * img.shape[0], *subimg_shape))


def calc_std_from_ratio(binary_img: np.ndarray, ratio):
    """Calculates the standard deviation of the subimages of an image, divided by a certain ratio."""
    divided_img = divide_img_to_subimages(binary_img, ratio)
    along_axis = tuple(np.arange(1, len(binary_img.shape)))
    ddof = 1  # for unbiased std
    return np.std(np.mean(divided_img, axis=along_axis), ddof=ddof)


def image_stats(
    img: np.ndarray, pf: float, ratios, z_score: float = 1.96
) -> list[float]:
    errs = []
    for ratio in ratios:
        std_ratio = calc_std_from_ratio(img, ratio)
        errs.append(100 * ((z_score * std_ratio) / pf))
    return errs


def ns_from_dims(img_dims, integral_range: float) -> list:
    n_dims = len(img_dims[0])
    den = integral_range**n_dims
    return [np.prod(np.array(i)) / den for i in img_dims]


def bernouli(pf: float, ns: list[int], conf: float = 0.95) -> np.ndarray:
    errs = []
    for n in ns:
        std_theo = ((1 / n) * (pf * (1 - pf))) ** 0.5
        errs.append(100 * (norm.interval(conf, scale=std_theo)[1] / pf))
    return np.array(errs, dtype=np.float64)


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


def fit_cls(err_exp, img_dims, pf, max_cls=150):
    err_exp = np.array(err_exp)
    cls = test_cls_set(err_exp, pf, np.arange(1, max_cls, 1), img_dims)
    cls = test_cls_set(err_exp, pf, np.linspace(cls - 1, cls + 1, 50), img_dims)
    # print(f'real cls = {cls}')
    return cls


def stat_analysis_error_classic(
    binary_img: np.ndarray, pf: float
):  # TODO see if to delete this or not
    ratios = [2**i for i in np.arange(1, int(np.log2(binary_img.shape[1])) - 5)]
    ratios.reverse()
    if binary_img.shape[0] > 1:
        ratios.append(1)
    ratios = ratios[-4:]
    edge_lengths = [binary_img.shape[1] // r for r in ratios]
    img_dims = [np.array((l,) * (len(binary_img.shape) - 1)) for l in edge_lengths]
    err_exp = image_stats(binary_img, pf, ratios)
    real_cls = fit_cls(err_exp, img_dims, pf)
    # TODO different size image 1000 vs 1500
    return real_cls


def pred_cls_is_off(pred_cls: float, binary_img: np.ndarray, pf: float):
    if pred_cls < 1:
        return True, 1
    one_im_stat_pred = stat_analysis_error_classic(binary_img, pf)
    if one_im_stat_pred > 1:  # could be erroneous stat. analysis prediction
        if pred_cls / one_im_stat_pred < 2 / 3:
            return True, 1
        if pred_cls / one_im_stat_pred > 2:
            return True, -1
    return False, 0


def change_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape, sign):
    """Changes the tpc function to be more positive or more negative, compared
    to the fast stat. analysis cls pred. of the single img."""
    if sign > 0:
        negatives = np.where(tpc - pf_squared < 0)
        tpc[negatives] += (pf_squared - tpc[negatives]) / 10
    else:
        positives = np.where(tpc - pf_squared > 0)
        tpc[positives] -= (tpc[positives] - pf_squared) / 10
    pred_cls = calc_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape)
    return tpc, pred_cls


def tpc_to_cls(tpc: np.ndarray, binary_image: np.ndarray) -> float:
    """Calculates the integral range from the tpc function."""

    img_shape = binary_image.shape
    middle_idx = np.array(tpc.shape) // 2
    pf = tpc[tuple(map(slice, middle_idx, middle_idx + 1))][0][0]
    print(pf, tpc.shape)
    dist_arr_before = np.indices(tpc.shape)
    dist_arr_before = np.abs((dist_arr_before.T - middle_idx.T).T)
    img_volume = np.prod(img_shape)
    # normalising the tpc s.t. different vectors would have different weights,
    # According to their volumes.
    norm_vol = (np.array(img_shape).T - dist_arr_before.T).T
    norm_vol = np.prod(norm_vol, axis=0) / img_volume
    dist_arr: np.ndarray = np.sqrt(np.sum(dist_arr_before**2, axis=0))
    end_dist = find_end_dist_tpc(pf, tpc, dist_arr)
    print(f"end dist = {end_dist}")
    pf_squared_end = np.mean(tpc[(dist_arr >= end_dist - 10) & (dist_arr <= end_dist)])

    pf_squared = (pf_squared_end + pf**2) / 2
    bool_array = dist_arr < end_dist

    # calculate the coefficient for the cls prediction:
    coeff = calc_coeff_for_cls_prediction(
        norm_vol, dist_arr, end_dist, int(img_volume), bool_array
    )
    pred_cls = calc_pred_cls(coeff, tpc, pf, pf_squared, bool_array, img_shape)
    pred_is_off, sign = pred_cls_is_off(pred_cls, binary_image, pf)
    while pred_is_off:
        how_off = "negative" if sign > 0 else "positive"
        print(f"pred cls = {pred_cls} is too {how_off}, CHANGING TPC VALUES")
        tpc, pred_cls = change_pred_cls(
            coeff, tpc, pf, pf_squared, bool_array, img_shape, sign
        )
        pred_is_off, sign = pred_cls_is_off(pred_cls, binary_image, pf)
    return pred_cls


def fit_to_errs_function(dim: int, n_voxels: int, a: float, b: float) -> float:
    return a / n_voxels**b


def get_std_model(dim: int, n_voxels: int) -> float:
    popt = {"2d": [48.20175315, 0.4297919], "3d": [444.803518, 0.436974444]}
    return fit_to_errs_function(dim, n_voxels, *popt[f"{dim}d"])


def normal_dist(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def get_prediction_interval(
    image_pf, pred_std, pred_std_error_std, conf_level=0.95, n_divisions=101
):
    """Get the prediction interval for the phase fraction of the material given the image phase
    fraction, the predicted standard deviation and the standard deviation of the prediction error.
    """
    # have a large enough number of stds to converge to 0 at both ends,
    # but not too large to make the calculation slow:
    std_dist_std = pred_std * pred_std_error_std  # TODO see if this fits
    num_stds = min(pred_std / std_dist_std - pred_std / std_dist_std / 10, 6)
    # First, make the "weights" or "error" distribution, the normal distribution of the stds
    # where the prediction std is the mean of this distribution:
    x_std_dist_bounds = [
        pred_std - num_stds * std_dist_std,
        pred_std + num_stds * std_dist_std,
    ]
    x_std_dist: np.ndarray = np.linspace(
        *x_std_dist_bounds, n_divisions
    )  # type: ignore
    std_dist = normal_dist(x_std_dist, mean=pred_std, std=std_dist_std)
    # Next, make the pf distributions, each row correspond to a different std, with
    # the same mean (observed pf) but different stds (x_std_dist), multiplied by the
    # weights distribution (std_dist).
    pf_locs = np.ones((n_divisions, n_divisions)) * image_pf
    pf_x_bounds = [image_pf - num_stds * pred_std, image_pf + num_stds * pred_std]
    pf_x_1d: np.ndarray = np.linspace(*pf_x_bounds, n_divisions)  # type: ignore
    pf_mesh, std_mesh = np.meshgrid(pf_x_1d, x_std_dist)
    # Before normalising by weight:
    pf_dist_before_norm = normal_dist(pf_mesh, mean=pf_locs, std=std_mesh)
    # Normalise by weight:
    pf_dist = (pf_dist_before_norm.T * std_dist).T
    # Sum the distributions over the different stds
    # print(np.sum(pf_dist, axis=0).shape, np.diff(x_std_dist).shape)
    print(np.diff(x_std_dist))
    sum_dist_norm = np.sum(pf_dist, axis=0) * np.diff(x_std_dist)[0]  # [0]
    # need a bit of normalization for symmetric bounds (it's very close to 1 already)
    sum_dist_norm /= np.trapz(sum_dist_norm, pf_x_1d)
    # Find the alpha confidence bounds
    cum_sum_sum_dist_norm = np.cumsum(sum_dist_norm * np.diff(pf_x_1d)[0])
    half_conf_level = (1 + conf_level) / 2
    conf_level_beginning = np.where(cum_sum_sum_dist_norm > 1 - half_conf_level)[0][0]
    conf_level_end = np.where(cum_sum_sum_dist_norm > half_conf_level)[0][0]
    # TODO: Check
    # if conf_level_end[0].size == 0:
    #    conf_level_end = -1

    # Calculate the interval
    return pf_x_1d[conf_level_beginning], pf_x_1d[conf_level_end]


def find_n_for_err_targ(
    n, image_pf, pred_std_error_std, err_target, conf_level=0.95, n_divisions=101
) -> float:
    std_bern = ((1 / n) * (image_pf * (1 - image_pf))) ** 0.5
    pred_interval = get_prediction_interval(
        image_pf, std_bern, pred_std_error_std, conf_level, n_divisions
    )
    err_for_img = image_pf - pred_interval[0]
    return (err_target - err_for_img) ** 2


def dims_from_n(n, equal_shape: bool, cls, dims):
    den = cls**dims
    if equal_shape:
        return (n * den) ** (1 / dims)
    else:
        # if dims == len(shape):
        raise ValueError("cannot define all the dimensions")
        # if len(shape) == 1:
        #    return ((n * den) / (shape[0] + cls - 1)) ** (1 / (dims - 1)) - cls + 1
        # else:
        #    return ((n * den) / ((shape[0] + cls - 1) * (shape[1] + cls - 1))) - cls + 1


def make_error_prediction(
    binary_img: np.ndarray,
    confidence: float = 0.95,
    target_error: float = 0.05,
    equal_shape: bool = True,
    model_error: bool = True,
) -> dict:
    phase_fraction = np.mean(binary_img)
    n_dims = len(binary_img.shape)  # 2D or 3D
    n_elems = int(np.prod(binary_img.shape))

    print("bpe0")
    two_point_correlation = radial_tpc(binary_img, n_dims == 3, True)
    print("bpe1")
    integral_range = tpc_to_cls(
        two_point_correlation,
        binary_img,
    )
    print("bpe2")

    n = ns_from_dims([np.array(binary_img.shape)], integral_range)
    print("bpe3")

    std_bern = (
        (1 / n[0]) * (phase_fraction * (1 - phase_fraction))
    ) ** 0.5  # TODO: this is the std of phi relative to Phi with
    std_model = get_std_model(n_dims, n_elems)
    abs_err_target = target_error * phase_fraction
    print("bpe4")
    if model_error:
        # calculate the absolute error for the image:
        conf_bounds = get_prediction_interval(
            phase_fraction, std_bern, std_model, confidence
        )
        print("bpe5")
        abs_err_for_img = phase_fraction - conf_bounds[0]
        # calculate the n for the error target:
        args = (phase_fraction, std_model, target_error, confidence)
        n_for_err_targ = minimize(find_n_for_err_targ, n, args=args)
        n_for_err_targ = n_for_err_targ.x[0]
        print("bpe6")
    else:  # TODO what is this useful for.. for when you trust the model completely?
        z = norm.interval(confidence)[1]
        abs_err_for_img = z * std_bern
        n_for_err_targ = (
            phase_fraction * (1 - phase_fraction) * (z / abs_err_target) ** 2
        )
    print("bpe7")
    l_for_err_targ = dims_from_n(n_for_err_targ, equal_shape, integral_range, n_dims)
    percentage_err_for_img = abs_err_for_img / phase_fraction

    if n_dims == 3:
        integral_range = integral_range[0]
        l_for_err_targ = l_for_err_targ[0]
        percentage_err_for_img = percentage_err_for_img[0]
        abs_err_for_img = abs_err_for_img[0]

    result = {
        "integral_range": integral_range,
        "z": z,
        "percent_err": percentage_err_for_img,
        "abs_err": abs_err_for_img,
        "l": l_for_err_targ,
    }
    return result  # percentage_err_for_img, l_for_err_targ, integral_range
