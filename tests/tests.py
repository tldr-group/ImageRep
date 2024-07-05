import numpy as np

np.random.seed(0)
from tifffile import imread

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

    def test_std_of_subimgs(self):
        test_pf = 0.3
        test_arr = np.random.binomial(1, test_pf, (400, 400))
        print(model.calc_std_from_ratio(test_arr, 2))


if __name__ == "__main__":
    unittest.main(argv=["example"])
"""
inp = imread("seg_stack.tiff")
if inp.shape[0] == 1:
    inp = inp[0, :, :]
print(np.unique(inp))
# 170 for default
microstructure = np.where(inp == 127, 1, 0)

make_error_prediction(microstructure, model_error=False)
"""
