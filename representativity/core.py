import numpy as np
from itertools import product, chain
from scipy.stats import norm
from scipy.optimize import minimize


# %% ======================== TWO-POINT CORRELATION METHODS ========================


def autocorrelation_orthant(
    binary_img: np.ndarray, num_elements: int, n_dims: int, desired_length: int = 100
) -> np.ndarray:
    """Calculates the autocorrelation function of a binary image using the FFT method
    for a single orthant, in all vectors right and down from the origin. Using the FFT
    computes the TPC for all these vectors in a single pass, instead of explicitly
    shifting the image by every vector r in [(0, 1) ... (0, l) .. (l, l)] and taking
    the product. This reduces the number of operations for an image of size N from
    N^2 (N shifts * N for each product) -> Nlog(N). We slice to $desired_length
    after to speed up calculations later.

    Note Two-Point Correlation (TPC) = autocorrelation and are used interchangably.

    [1] 'Microstructure sensitive design for performance optimization', 2012
    Adams, Brent L and Kalidindi, Surya R and Fullwood, David T
    [2] 'Efficient generation of anisotropic N-field microstructures
    from 2-point statistics using multi-output Gaussian random fields', 2022
    Robertson, Andreas E and Kalidindi, Surya R

    :param binary_img: 2/3D binary arr for the microstructure
    :type binary_img: np.ndarray
    :param num_elements: total N elements in microstructure
    :type num_elements: int
    :param n_dims: whether arr is 2/3D
    :type n_dims: int
    :param desired_length: post TPC crop length, defaults to 100
    :type desired_length: int, optional
    :return:  2/3D array for orthant where tpc[y, x] is the tpc for the vector (y,x)
    :rtype: np.nddary
    """
    ax = list(range(0, len(binary_img.shape)))
    img_FFT = np.fft.rfftn(binary_img, axes=ax)
    tpc = (
        np.fft.irfftn(img_FFT.conjugate() * img_FFT, s=binary_img.shape, axes=ax).real
        / num_elements
    )
    # multidimensional slicing up to desired length i.e tpc[:100, :100, :100]
    return tpc[tuple(map(slice, (desired_length,) * n_dims))]


def two_point_correlation_orthant(
    binary_img: np.ndarray,
    n_dims: int,
    desired_length: int = 100,
    periodic: bool = True,
) -> np.ndarray:
    """Calculates the two point correlation function of an image along an orthant.
    If we do not want to use periodic boundary conditions for calculating the autocorrelation,
    it pads the image with $desired_length number of zeros before calculating the 2PC function
    using the FFT method. After the FFT calculation, it normalises the result by the number of
    possible occurences of the 2PC function.

    Periodic BCs are better for stability, so not using them (i.e not periodic == True) is uncommon.

    :param binary_img: 2/3D binary arr for the microstructure
    :type binary_img: np.ndarray
    :param n_dims: whether arr is 2/3D
    :type n_dims: int
    :param desired_length: post TPC crop length, defaults to 100
    :type desired_length: int, optional
    :param periodic: whether periodic BCs applied, defaults to True
    :type periodic: bool, optional
    :return: (normalised) tpc for a orthant
    :rtype: np.ndarray
    """
    img_shape = binary_img.shape
    normalisation_for_tpc: np.ndarray
    if not periodic:  # padding the image with zeros
        indices_img = np.indices(img_shape) + 1

        normalisation_for_tpc = np.flip(np.prod(indices_img, axis=0))
        # normalisation_for_tpc is an arr where the entry arr[x, y] counts the number of the original entries of img that
        # will be present after shifting by x,y i.e for a shift 0,0 this is mag(img)
        # this lets you normalise the mean with the 'ghost dimensions' later for non-periodic images
        binary_img = np.pad(
            binary_img, [(0, desired_length) for _ in range(n_dims)], "constant"
        )

    num_elements = int(np.product(img_shape))
    # 2D tpc array up to $desired_legth * $desired_length
    tpc_desired = autocorrelation_orthant(
        binary_img, num_elements, n_dims, desired_length
    )

    if not periodic:
        # normalising the result as we have more 0s than would otherwise have
        normalisation_for_tpc = normalisation_for_tpc[
            tuple(map(slice, tpc_desired.shape))
        ]  # multidimensional slicing
        normalisation_for_tpc = num_elements / normalisation_for_tpc
        # normalisation_for_tpc is an array of adjustments applied pointwise to the tpc_desired
        return normalisation_for_tpc * tpc_desired
    else:
        return tpc_desired


def two_point_correlation(
    binary_img: np.ndarray,
    desired_length: int = 100,
    volumetric: bool = False,
    periodic: bool = True,
) -> np.ndarray:
    """Compute TPC for all orthants (ND quadrants) in the input data. These orthants are related to
    directions of vectors we would use to shift the image before computing the self-product were we
    not using the FFT approach. They correspond to the axes of the image that are flipped before
    performing the FFT. The orthant TPCs are then recombined into $result

    :param binary_img: 2/3D binary arr for the microstructure
    :type binary_img: np.ndarray
    :param desired_length: post TPC crop length, defaults to 100
    :type desired_length: int, optional
    :param volumetric: if 3D or 2D, defaults to False
    :type volumetric: bool, optional
    :param periodic: whether periodic BCs applied, defaults to True
    :type periodic: bool, optional
    :return: a 2/3D array composed of orthant-wise tpcs. For a 2D array this is
    (2 * $desired_length, 2 * $desired_length).
    :rtype: np.ndarray
    """
    n_dims = 3 if volumetric else 2
    # orthant = N-D quadrant. stored here, indexed by axis
    orthants: dict[tuple, np.ndarray] = {}
    # calculating the 2PC function for each orthant, saving the result in a dictionary
    for axis in product((1, 0), repeat=n_dims - 1):
        # flip list is just a list of axes to flip, sometimes no flip happens
        # computing tpc of left and down img = computing tpc of left and up flipped (T->B) image
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
        # symmetry flip
        orthants[opposite_axis] = np.flip(original_tpc_orthant)

    # result is our 2/3D array of TPC values up to desired length (in half-manhattan distance NOT euclidean)
    result = np.zeros((desired_length * 2 + 1,) * n_dims)
    for axis in orthants.keys():
        # axis looks like (1, 1)
        axis_idx = np.array(axis) * desired_length
        # axis_idx looks like (100, 100)
        # slice to input: mapping of orthant axis to location in result i.e [0:100, 0:100]
        slice_to_input = tuple(
            map(slice, axis_idx, axis_idx + desired_length)
        )  # TODO: check with A, this used to be + 1
        result[slice_to_input] = orthants[axis]
    return result


def radial_tpc(
    binary_img: np.ndarray, volumetric: bool = False, periodic: bool = True
) -> np.ndarray:
    """TPC entrypoint"""
    # this is a problem where arr not square, should take minimum of dimension (for now)
    # TODO: make desired length different in all dimensions
    # TODO: does this need to be its own function that takes all the same arguments as two_point_...
    img_y_length: int = min(binary_img.shape)  # binary_img.shape[0]
    # img_y_length: int = binary_img.shape[0]
    print(f"desired length: {img_y_length}, shape: {binary_img.shape}")
    # desired length: dims of output of fft arr,
    desired_length = (img_y_length // 2) if periodic else (img_y_length - 1)
    return two_point_correlation(
        binary_img,
        desired_length=desired_length,
        volumetric=volumetric,
        periodic=periodic,
    )


# %% ======================== CHARACTERISTIC LENGTH SCALE METHODS ========================


def calc_coeff_for_cls_prediction(
    norm_vol: np.ndarray,
    dist_arr: np.ndarray,
    end_dist: float,
    img_volume: int,
    bool_array: np.ndarray,
) -> float:

    # norm vol: normilisation volumes
    # dist_arr still euclidean dists of orthant indices from centre
    # end_dist: when tpc stops fluctuating
    # bool arr: distances less than end_distance
    # looking for C_r0 here

    # sum of normalisations of all vectors less than r_0/end_dist
    # sum_of_small_radii = np.sum(norm_vol[dist_arr < end_dist])
    sum_of_small_radii = np.sum(norm_vol[bool_array])
    coeff_1 = img_volume / (img_volume - sum_of_small_radii)
    # TODO: check if np.sum(norm_vol[bool_array]) == np.sum(norm_vol[dist_arr < end_dist])
    # TODO: remove end dist and dist_arr args
    # coeff_2 = (1 / img_volume) * (np.sum(bool_array) - np.sum(norm_vol[bool_array]))
    coeff_2 = (1 / img_volume) * (np.sum(bool_array) - sum_of_small_radii)

    coeff_product = coeff_1 * coeff_2
    while coeff_product > 1:
        print(f"coeff product = {coeff_product}")
        coeff_product /= 1.1
    # output is effectively (but not exactly) C_r0
    return coeff_1 / (1 - coeff_product)


def find_end_dist_idx(
    pf: float, tpc: np.ndarray, dist_arr: np.ndarray, distances: np.ndarray
):
    """Finds the distance before the tpc function plateaus."""
    # looking at change in TPC radially from centre
    # deviation = how different TPC is from image phase fraction ^2
    # because TPC should converge to real phase fraction squared

    # looking at percentage of all TPCs in the ring that are outside
    # of 5% of the image phase fraction squared
    # => where TPC stops fluctuating

    # the rings are all distances from (0, 100) then (100, 200) then ,,,
    # based on distances
    # TODO: renamed distances -> r0_bounds or ring_distances etc

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


def find_end_dist_tpc(
    phase_fraction: float, tpc: np.ndarray, dist_arr: np.ndarray
) -> float:
    # assumption is image is at least 200 in every dimensoom
    # TODO: signpost in paper?
    max_img_dim = np.max(dist_arr)
    if max_img_dim < 200:
        print(f"Max img dim of {max_img_dim} < 200px, using small method")
        # this gives more unstable results but works for smaller images
        distances = np.linspace(0, int(max_img_dim), 100)
    else:
        # this is the correct way as it reduces number of operations (but fails for small images)
        distances = np.concatenate([np.arange(0, int(max_img_dim), 100)])

    # check the tpc change and the comparison to pf^2
    # over bigger and bigger discs:
    return find_end_dist_idx(phase_fraction, tpc, dist_arr, distances)


def calc_pred_cls(
    coeff: float,
    tpc: np.ndarray,
    pf: float,
    pf_squared: float,
    bool_array: np.ndarray,
    im_shape: tuple[int, ...],
) -> float:
    # second term is integral of tpc - pf_squared
    # eq 11 in paper (for now)
    # NB don't need |X_r|/|X| norm in summand as in \psi in eq (9) as already
    # taken care of due to periodicity and coeffs found previously
    # TODO: check
    pred_cls = coeff / (pf - pf_squared) * np.sum(tpc[bool_array] - pf_squared)
    # this goes from length^N -> length to get a length scale
    if pred_cls > 0:
        pred_cls = pred_cls ** (1 / 3) if len(im_shape) == 3 else pred_cls ** (1 / 2)
    return pred_cls


def divide_img_to_subimages(img: np.ndarray, subimg_ratio) -> np.ndarray:
    """Divides an image to non-overlapping subimages from a certain ratio."""
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
    # for each of the edge length rations, calc std of phase fraction of
    #  subimages of size image.shape / ratio
    errs = []  # std_err of phase fraction
    for ratio in ratios:
        std_ratio = calc_std_from_ratio(img, ratio)
        errs.append(100 * ((z_score * std_ratio) / pf))
    return errs


def ns_from_dims(img_dims, integral_range: float) -> list:
    # translation from cls -> ns = (number of samples) from your
    # bernoulli distribution with feature size cls

    # img dims is a list of image dimensions i.e [(h1, w1), (h2, w2)]
    # from our sub images
    n_dims = len(img_dims[0])
    # den = denominator
    den = integral_range**n_dims
    # subimage (hyper)volume / integral range (hyper) volume
    return [np.prod(np.array(i)) / den for i in img_dims]


def bernouli(pf: float, ns: list[int], conf: float = 0.95) -> np.ndarray:
    errs = []
    for n in ns:
        std_theo = ((1 / n) * (pf * (1 - pf))) ** 0.5
        errs.append(100 * (norm.interval(conf, scale=std_theo)[1] / pf))
    return np.array(errs, dtype=np.float64)


def test_cls_set(err_exp, pf, clss, img_dims):
    # test all the clses in clss (=cls set)
    # err exp is error from the pfs of the sub images, we compare to the
    # ideal/theoretical of the bernoulli of the image divided into cls (hyoer)cubes
    err_fit = []
    for cls in clss:
        ns = ns_from_dims(img_dims, cls)
        # given that the cls is correct, this is the error in the standard statistical method
        err_model = bernouli(pf, ns)
        difference = abs(err_exp - err_model)
        err = np.mean(difference)
        err_fit.append(err)
    cls = clss[np.argmin(err_fit)].item()
    return cls


def fit_cls(err_exp, img_dims, pf, max_cls=150):
    # find the cls that best explains the phase fraction std errors from
    # the various subimages i.e that best aligns with our bernoulli assumption
    # which holds if the features are finite
    err_exp = np.array(err_exp)
    # coarse scan
    cls = test_cls_set(err_exp, pf, np.arange(1, max_cls, 1), img_dims)
    # fine scan
    cls = test_cls_set(err_exp, pf, np.linspace(cls - 1, cls + 1, 50), img_dims)
    # print(f'real cls = {cls}')
    return cls


def stat_analysis_error_classic(
    binary_img: np.ndarray, pf: float
):  # TODO see if to delete this or not

    # crop in increasing powers of 2
    # TODO: binary_img.shape[1] is always one dimensions, should be the smallest dimension
    # or dimension specific
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
    # can compare our predicted cls to the standard way of calculating the cls
    # standard way = taking different crops of the image then calculating the
    # std deviation

    # cls in pixel length
    if pred_cls < 1:
        return True, 1
    # one image statistical/classical prediction of the cls
    one_im_stat_pred = stat_analysis_error_classic(binary_img, pf)
    if one_im_stat_pred > 1:  # could be erroneous stat. analysis prediction
        # if pred cls too low or too high compared to statistical method,
        # return true and the direction of the error (1 for too low, -1 for too high)
        if pred_cls / one_im_stat_pred < 2 / 3:
            return True, 1
        if pred_cls / one_im_stat_pred > 2:
            return True, -1
    return False, 0


def change_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape, sign):
    """Changes the tpc function to be more positive or more negative, compared
    to the fast stat. analysis cls pred. of the single img."""
    # rationale is that changing the tpcs and then predicting is more
    # likely to reutrn a 'good' cls than just snapping to the error
    # bound regions of [(2/3) * stat_pred, 2 * stat_pred]

    # pf_squared = measured image pf ^2

    if sign > 0:
        negatives = np.where(tpc - pf_squared < 0)
        tpc[negatives] += (pf_squared - tpc[negatives]) / 10
    else:
        positives = np.where(tpc - pf_squared > 0)
        tpc[positives] -= (tpc[positives] - pf_squared) / 10
    pred_cls = calc_pred_cls(coeff, tpc, pf, pf_squared, bool_array, im_shape)
    return tpc, pred_cls


def tpc_to_cls(tpc: np.ndarray, binary_image: np.ndarray) -> float:
    """Compute the Characteristic Length Scale (CLS) from the TPC array and microstructure.
    First, the distance where the TPC stops fluctuating (r_0) is found. Using this the TPC-estimated
    phase fraction is found. The model from the paper (sections 4.1 and 4.2) is used to calculate
    the coefficients. All these quantities are then used to calculate the CLS. This TPC-predicted
    CLS is compared to the classical (statistical) method for CLS estimation and adjusted if
    less than 2/3x or more than 2X of CLS_stats.

    NB: no guarantee end_dist < our $desired_length

    :param tpc: 2D array of orthant TPCs, shape (2*$desired_length, 2*$desired_length) in 2D
    :type tpc: np.ndarray
    :param binary_image: 2/3D binary arr for the microstructure
    :type binary_image: np.ndarray
    :return: CLS of the binary microstructure (roughly the feature size)
    :rtype: float
    """

    img_shape = binary_image.shape
    # the middle index is the 0,0 TPC because of how the TPC function laid them out
    middle_idx = np.array(tpc.shape) // 2
    # this is the measured image phase fraction (i.e the product of the image with itself)
    image_phase_fraction = tpc[
        tuple(map(slice, middle_idx, middle_idx + 1))
    ].item()  # [0][0]
    # 'raw_dist_arr' = before taking sqrts
    raw_dist_arr = np.indices(tpc.shape)
    # arr indices != coordinates, we care about distances from centre of coords so remap
    remapped_dist_arr = np.abs((raw_dist_arr.T - middle_idx.T).T)
    img_volume = np.prod(img_shape)
    # normalising the tpc s.t. different vectors would have different weights,
    # According to their volumes.
    # number of r s.t x + r \in x i.e same as other normaliser
    norm_vol = (np.array(img_shape).T - remapped_dist_arr.T).T
    norm_vol = np.prod(norm_vol, axis=0) / img_volume
    # euclidean distances
    euc_dist_arr: np.ndarray = np.sqrt(np.sum(remapped_dist_arr**2, axis=0))
    # end dist = r0 = dist when tpc stops fluctuating
    end_dist = find_end_dist_tpc(image_phase_fraction, tpc, euc_dist_arr)

    # take mean of tpcs in the outer ring width 10 from end dist
    tpc_phase_fraction_squared = np.mean(
        tpc[(euc_dist_arr >= end_dist - 10) & (euc_dist_arr <= end_dist)]
    )
    # take mean of this estimated pf_square from tpc and measured pf_squared from image
    # emprical result - broadly speaking mean is more stable
    phase_fraction_squared = (tpc_phase_fraction_squared + image_phase_fraction**2) / 2
    bool_array = euc_dist_arr < end_dist

    # calculate the coefficient needed for the cls prediction
    coeff = calc_coeff_for_cls_prediction(
        norm_vol, euc_dist_arr, end_dist, int(img_volume), bool_array
    )

    pred_cls = calc_pred_cls(
        coeff, tpc, image_phase_fraction, phase_fraction_squared, bool_array, img_shape
    )
    # use classical apporach to see if CLS is off
    pred_is_off, sign = pred_cls_is_off(pred_cls, binary_image, image_phase_fraction)
    while pred_is_off:
        how_off = "negative" if sign > 0 else "positive"
        print(f"pred cls = {pred_cls} is too {how_off}, CHANGING TPC VALUES")
        tpc, pred_cls = change_pred_cls(
            coeff,
            tpc,
            image_phase_fraction,
            phase_fraction_squared,
            bool_array,
            img_shape,
            sign,
        )
        pred_is_off, sign = pred_cls_is_off(
            pred_cls, binary_image, image_phase_fraction
        )
    return pred_cls


def fit_to_errs_function(dim: int, n_voxels: int, a: float, b: float) -> float:
    return a / n_voxels**b


def get_std_model(dim: int, n_voxels: int) -> float:
    # fit from microlib
    popt = {"2d": [48.20175315, 0.4297919], "3d": [444.803518, 0.436974444]}
    return fit_to_errs_function(dim, n_voxels, *popt[f"{dim}d"])


def normal_dist(
    x: np.ndarray, mean: float | np.ndarray, std: float | np.ndarray
) -> np.ndarray:
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def get_prediction_interval(
    image_pf, pred_std, pred_std_error_std, conf_level=0.95, n_divisions=101
) -> tuple[float, float]:
    """Get the prediction interval for the phase fraction of the material given the image phase
    fraction, the predicted standard deviation and the standard deviation of the prediction error.
    """
    # have a large enough number of stds to converge to 0 at both ends,
    # but not too large to make the calculation slow:
    std_dist_std = pred_std * pred_std_error_std  # TODO see if this fits
    num_stds = min(pred_std / std_dist_std - pred_std / std_dist_std / 10, 6)
    # First, make the "weights" or "error" distribution, the normal distribution of the stds
    # where the prediction std is the mean of this distribution:
    x_std_dist_bounds = (
        pred_std - num_stds * std_dist_std,
        pred_std + num_stds * std_dist_std,
    )
    x_std_dist: np.ndarray = np.linspace(*x_std_dist_bounds, n_divisions)
    std_dist = normal_dist(x_std_dist, mean=pred_std, std=std_dist_std)
    # Next, make the pf distributions, each row correspond to a different std, with
    # the same mean (observed pf) but different stds (x_std_dist), multiplied by the
    # weights distribution (std_dist).
    pf_locs = np.ones((n_divisions, n_divisions)) * image_pf
    pf_x_bounds = (image_pf - num_stds * pred_std, image_pf + num_stds * pred_std)
    pf_x_1d: np.ndarray = np.linspace(*pf_x_bounds, n_divisions)
    pf_mesh, std_mesh = np.meshgrid(pf_x_1d, x_std_dist)
    # Before normalising by weight:
    pf_dist_before_norm = normal_dist(pf_mesh, mean=pf_locs, std=std_mesh)
    # Normalise by weight:
    pf_dist = (pf_dist_before_norm.T * std_dist).T
    # Sum the distributions over the different stds
    print(pf_dist.shape, x_std_dist.shape)
    # print(np.sum(pf_dist, axis=0).shape, np.diff(x_std_dist).shape)
    sum_dist_norm = np.sum(pf_dist, axis=0) * np.diff(x_std_dist)  # [0]  # [0]
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

    two_point_correlation = radial_tpc(binary_img, n_dims == 3, True)
    integral_range = tpc_to_cls(
        two_point_correlation,
        binary_img,
    )

    n = ns_from_dims([np.array(binary_img.shape)], integral_range)
    # bern = bernouilli
    std_bern = (
        (1 / n[0]) * (phase_fraction * (1 - phase_fraction))
    ) ** 0.5  # TODO: this is the std of phi relative to Phi with
    std_model = get_std_model(n_dims, n_elems)
    abs_err_target = target_error * phase_fraction
    if model_error:
        # calculate the absolute error for the image:
        conf_bounds = get_prediction_interval(
            phase_fraction, std_bern, std_model, confidence
        )
        abs_err_for_img = phase_fraction - conf_bounds[0]
        # calculate the n for the error target:
        args = (phase_fraction, std_model, target_error, confidence)
        n_for_err_targ = minimize(find_n_for_err_targ, n, args=args)
        n_for_err_targ = n_for_err_targ.x[0]
    else:  # TODO what is this useful for.. for when you trust the model completely?
        z = norm.interval(confidence)[1]
        abs_err_for_img = z * std_bern
        n_for_err_targ = (
            phase_fraction * (1 - phase_fraction) * (z / abs_err_target) ** 2
        )
    l_for_err_targ = dims_from_n(n_for_err_targ, equal_shape, integral_range, n_dims)
    percentage_err_for_img = abs_err_for_img / phase_fraction

    # if n_dims == 3:
    # integral_range = integral_range[0]
    # l_for_err_targ = l_for_err_targ[0]
    # percentage_err_for_img = percentage_err_for_img[0]
    # abs_err_for_img = abs_err_for_img[0]

    result = {
        "integral_range": integral_range,
        "z": z,
        "percent_err": percentage_err_for_img,
        "abs_err": abs_err_for_img,
        "l": l_for_err_targ,
    }
    return result  # percentage_err_for_img, l_for_err_targ, integral_range
