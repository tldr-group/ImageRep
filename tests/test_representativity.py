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
    img = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
    tpc = util.two_point_correlation(img, desired_length=2, periodic=False, threed=False)
    assert np.allclose(tpc, np.array([[0.   , 0.   , 3/9, 0.   , 1.   ],
       [0.5  , 0.25 , 1/6, 0.5  , 0.5  ],
       [3/9, 1/6, 5/9, 1/6, 3/9],
       [0.5  , 0.5  , 1/6, 0.25 , 0.5  ],
       [1.   , 0.   , 3/9, 0.   , 0.   ]]))
    
def test_2pc_2d_periodic():
    """Test the two point correlation function for a non-periodic 2D image"""
    img = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
    tpc = util.two_point_correlation(img, desired_length=2, periodic=True, threed=False)
    assert np.allclose(tpc, np.array([[2/9, 4/9, 2/9, 2/9, 4/9],
       [4/9, 2/9, 2/9, 4/9, 2/9],
       [2/9, 2/9, 5/9, 2/9, 2/9],
       [2/9, 4/9, 2/9, 2/9, 4/9],
       [4/9, 2/9, 2/9, 4/9, 2/9]]))

# TODO for tpc, needs periodic 2D image, non-periodic 3D image, and periodic 3D image
