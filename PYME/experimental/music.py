import numpy as np
from scipy import ndimage

def remove_fixed_pattern(data, axis=1, eps=0.1):
    """
    Estimate and remove fixed pattern gain from a 2D image

    Algorithm operates by finding the scaling factor for each row/column
    that makes the data in each pixel as close as possible to the
    mean of the neighbouring pixels.
    """

    #neighbour_mean = ndimage.convolve1d(data, [0.5, 0, 0.5], axis=(1-axis))
    
    neighbour_mean = ndimage.gaussian_filter1d(data, 1, axis=(1-axis))
    
    scaling = neighbour_mean/(data + eps) 
    # take weighted average of scaling factor for each columm/row
    scaling = (data*scaling).sum(axis=axis)/(data.sum(axis=axis) + eps)

    if axis == 1:
        return data*scaling[:, np.newaxis]
    else:
        return data*scaling[np.newaxis, :]
    