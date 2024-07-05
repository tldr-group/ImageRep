import numpy as np
from tifffile import imread

from core import make_error_prediction

inp = imread("seg_stack.tiff")
if inp.shape[0] == 1:
    inp = inp[0, :, :]
print(np.unique(inp))
# 170 for default
microstructure = np.where(inp == 127, 1, 0)

make_error_prediction(microstructure, model_error=False)
