import pytest
import numpy as np
from representativity import util 
# from tests.utils import *
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
#  Testing the main solver


def test_2pc_2d_non_periodic():
    """Test the two point correlation function for a non-periodic 2D image"""
    img = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1]])
    tpc = util.two_point_correlation(img, desired_length=2, periodic=False, threed=False)
    assert np.allclose(tpc, np.array([[6/9, 1/6],[2/6, 2/4]]))

# TODO for tpc, needs periodic 2D image, non-periodic 3D image, and periodic 3D image
