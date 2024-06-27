import numpy as np
from itertools import product


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
) -> float:
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
    num_elements: int = int(np.product(img_shape))
    tpc_desired = autocorrelation_orthant(
        binary_img, num_elements, n_dims, desired_length
    )

    if not periodic:  # normalising the result
        normaliser = normaliser[tuple(map(slice, tpc_desired.shape))]
        normaliser = num_elements / normaliser
        return normaliser * tpc_desired
    else:
        return float(tpc_desired)


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
        slice_to_input = tuple(map(slice, axis_idx, axis_idx + desired_length + 1))
        result[slice_to_input] = orthants[axis]
    return result


def radial_tpc(
    binary_img: np.ndarray, volumetric: bool = False, periodic: bool = True
) -> np.ndarray:
    img_y_length: int = binary_img.shape[0]
    desired_length = (img_y_length // 2) if periodic else (img_y_length - 1)
    return two_point_correlation(
        binary_img,
        desired_length=desired_length,
        volumetric=volumetric,
        periodic=periodic,
    )


def make_error_prediction(
    binary_img: np.ndarray,
    confidence: float = 0.95,
    target_error: float = 0.05,
    equal_shape: bool = True,
) -> tuple[float, float, float]:
    phase_fraction = np.mean(binary_img)
    n_dims = len(binary_img.shape)  # 2D or 3D

    two_point_correlation = radial_tpc(binary_img, n_dims == 3, True)

    return False
