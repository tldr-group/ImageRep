import numpy as np

np.random.seed(0)
from skimage.draw import disk, rectangle
from tifffile import imread
import matplotlib.pyplot as plt

import representativity.core as model

import unittest

TEST_TPC_ARR = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
NON_PERIODIC_GT = np.array(
    [
        [0.0, 0.0, 3 / 9, 0.0, 1.0],
        [0.5, 0.25, 1 / 6, 0.5, 0.5],
        [3 / 9, 1 / 6, 5 / 9, 1 / 6, 3 / 9],
        [0.5, 0.5, 1 / 6, 0.25, 0.5],
        [1.0, 0.0, 3 / 9, 0.0, 0.0],
    ]
)

PERIODIC_GT = np.array(
    [
        [2 / 9, 4 / 9, 2 / 9, 2 / 9, 4 / 9],
        [4 / 9, 2 / 9, 2 / 9, 4 / 9, 2 / 9],
        [2 / 9, 2 / 9, 5 / 9, 2 / 9, 2 / 9],
        [2 / 9, 4 / 9, 2 / 9, 2 / 9, 4 / 9],
        [4 / 9, 2 / 9, 2 / 9, 4 / 9, 2 / 9],
    ]
)

DEFAULT_MICROSTRUCTURE = imread("tests/resources/default.tiff")[0]
phases = np.unique(DEFAULT_MICROSTRUCTURE)
DEFAULT_BINARY_IMG = np.where(DEFAULT_MICROSTRUCTURE == phases[1], 1, 0)


def post_process_tpc(padded_tpc: np.ndarray) -> np.ndarray:
    n_dims = len(padded_tpc.shape)
    volumetric = True if (n_dims == 3) else False
    # remove padding columns
    if volumetric:
        tpc = padded_tpc[:-1, :-1, :-1]
    else:
        tpc = padded_tpc[:-1, :-1]
    half_idx = tpc.shape[0] // 2
    for i in range(n_dims):
        tpc = np.delete(tpc, (half_idx), axis=i)
    return tpc


class UnitTests(unittest.TestCase):
    """Unit tests of component functions."""

    def test_2PC_on_binary_arrs(self):
        """Test TPC on 2/3D binary arrs, where their TPC should equal their value"""
        l = 5
        for n_dims in (2, 3):
            for val in (0, 1):
                shape = np.zeros(n_dims, dtype=np.uint8) + l
                img = np.zeros(shape) + val
                is_volumetric = True if n_dims > 2 else False
                padded_tpc = model.two_point_correlation(
                    img, desired_length=3, volumetric=is_volumetric, periodic=False
                )
                tpc = post_process_tpc(padded_tpc)
                assert np.allclose(tpc, img)

    def test_2PC_on_test_arr(self):
        """Test the two point correlation function for a periodic and non-periodic 2D image"""

        l = TEST_TPC_ARR.shape[0]  # usually 3
        for is_periodic, gt in zip((False, True), (NON_PERIODIC_GT, PERIODIC_GT)):
            padded_tpc = model.two_point_correlation(
                TEST_TPC_ARR, desired_length=l, volumetric=False, periodic=is_periodic
            )
            tpc = post_process_tpc(padded_tpc)
            assert np.allclose(tpc, gt)

    def test_subimg_splitter(self):
        for ratio in (2, 4, 8):
            subimgs = model.divide_img_to_subimages(DEFAULT_BINARY_IMG, ratio)
            assert len(subimgs) == ratio**2


class IntegrationTests(unittest.TestCase):
    """Integration tests of component functions."""

    def test_cls_disk(self):
        """Measure CLS of $n_disks in binary arr with radius $radius. CLS should be ~= 2*$radius."""
        print("## Test case: characteristic length scale on random disks")
        y, x, n_disks, radius = 500, 500, 40, 40
        arr = np.zeros((y, x), dtype=np.uint8)
        for i in range(n_disks):
            dx, dy = np.random.randint(0, y), np.random.randint(0, x)
            rr, cc = disk((dy, dx), radius, shape=(y, x))
            arr[rr, cc] = 1
        vf = np.mean(arr)
        tpc = model.radial_tpc(arr, False, False)
        integral_range = model.tpc_to_cls(tpc, arr)
        print(
            f"CLS={integral_range:.3f}, VF={vf:.3f} for {n_disks} disks with diameter {2 * radius} on {x}x{y} image \n"
        )
        # plt.imsave("disk.png", arr)
        assert np.isclose(integral_range, 2 * radius, rtol=0.05)

    def test_cls_squares(self):
        print(
            "## Test case: characteristic length scale on random squares of increasing size"
        )
        target_vf = 0.5
        y, x, l = 500, 500, 10
        n_rects = int((y / l) * target_vf) ** 2
        print(n_rects)
        arr = np.zeros((y, x), dtype=np.uint8)
        for i in range(n_rects):
            dx, dy = np.random.randint(0, y), np.random.randint(0, x)
            rr, cc = rectangle((dy, dx), extent=(l, l), shape=(y, x))
            arr[rr, cc] = 1

        tpc = model.radial_tpc(arr, False, False)
        integral_range = model.tpc_to_cls(tpc, arr)
        # plt.imsave("foo.png", arr)
        print(integral_range, np.sqrt(2) * l)

    def test_repr_pred(self):
        """Measure the representativity of a crop of our default microstructure, finding the image edge length
        needed to reach a given $desired_error. Then crop the microstructure to this (larger) edge length,
        measuring the representativity and percent error again. If our model is correct (and conservative),
         then this new refined percent error < predicted error from the first measurement.
        """
        print("## Test case: representativity estimation")
        desired_error = 0.1
        crop = DEFAULT_BINARY_IMG[:300, :300]
        result = model.make_error_prediction(
            crop, 0.95, desired_error, model_error=False
        )
        l_for_err = int(result["l"])
        print(
            f"Need edge length {l_for_err} for better than {desired_error:.3f}% phase fraction error, currently {result['percent_err']:.3f}%"
        )
        wider_crop = DEFAULT_BINARY_IMG[:l_for_err, :l_for_err]
        refined_result = model.make_error_prediction(
            wider_crop, 0.95, 0.05, model_error=False
        )
        print(
            f"{refined_result['percent_err']:.3f}% phase fraction error at l={l_for_err}\n"
        )
        assert refined_result["percent_err"] < result["percent_err"]


if __name__ == "__main__":
    unittest.main(argv=["example"])
