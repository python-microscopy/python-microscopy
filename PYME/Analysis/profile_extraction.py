"""

"""
import numpy as np
from scipy import ndimage

def extract_profile(image, x0, y0, x1, y1, width=1):
    """
    Extract a line profile from (x0, y0) to (x1, y1).
    
    Parameters
    ----------
    image :  2D ndarray
        the image to extract from
    x0, y0 : int
        coordinates of start of profile
    x1, y1 : int
        coordinates of profile end
    width : int
        profile width in pixels (number of parallel profiles to average over)

    Returns
    -------
    
    a line profile of the image, interpolated to match the pixel spacing

    """
    w = int(np.floor(0.5 * width))
    
    Dx = x1 - x0
    Dy = y1 - y0
    
    l = np.sqrt((Dx ** 2 + Dy ** 2))
    
    dx = Dx / l
    dy = Dy / l
    
    if Dx == 0 and Dy == 0: #special case - profile is orthogonal to current plane
        d_x = w
        d_y = w
    else:
        d_x = w * abs(dy)
        d_y = w * abs(dx)
    
    #pixel indices at 1-pixel spacing
    t = np.arange(np.ceil(l))
    
    x_0 = min(x0, x1)
    y_0 = min(y0, y1)
    
    d__x = abs(d_x) + 1
    d__y = abs(d_y) + 1
    
    #if (self.do.slice == self.do.SLICE_XY):
    ims = image[int(min(x0, x1) - d__x):int(max(x0, x1) + d__x + 1),
          int(min(y0, y1) - d__y):int(max(y0, y1) + d__y + 1)].squeeze()
    
    splf = ndimage.spline_filter(ims)
    
    p = np.zeros(len(t))
    
    x_c = t * dx + x0 - x_0
    y_c = t * dy + y0 - y_0
    
    for i in range(-w, w + 1):
        #print np.vstack([x_c + d__x +i*dy, y_c + d__y + i*dx])
        p += ndimage.map_coordinates(splf, np.vstack([x_c + d__x + i * dy, y_c + d__y - i * dx]),
                                     prefilter=False)
    
    p = p / (2 * w + 1)
    
    return p