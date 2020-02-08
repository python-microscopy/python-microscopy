import numpy as np
import threading
from .distHist import *

def distanceHistogramThreadWrapper(x1, y1, x2, y2, nBins, binSize, hist):
    # Wrapper to deal with capturing the threading result, because we want this to work 
    # with Python 2. If we used Python 3 only, we could use concurrent.futures.ThreadPoolExecutor
    # as an executor for the pool of threads and results.
    hist += distanceHistogram(x1, y1, x2, y2, nBins, binSize)

def distanceHistogramThreaded(x1, y1, x2, y2, nBins, binSize, split=12):
    """
    distHist.c distanceHistogram computed in # split parallel tasks
    """
    hist = np.zeros(nBins)  # the final histogram

    # Compute a series of histograms from points in parallel
    split = np.minimum(split, len(x1))
    split_indices = np.array_split(np.arange(len(x1)),split)
    threads = []
    for _i in np.arange(len(split_indices)):
        _si = split_indices[_i]
        t = threading.Thread(target=distanceHistogramThreadWrapper, args=(x1[_si], y1[_si], x2[_si[0]:], y2[_si[0]:], nBins, binSize, hist))
        threads.append(t)
        t.start()
    
    # Make sure all threads have executed
    for t in threads:
        t.join()

    return hist

def distanceHistogram3DThreadWrapper(x1, y1, z1, x2, y2, z2, nBins, binSize, hist):
    # Wrapper to deal with capturing the threading result
    hist += distanceHistogram3D(x1, y1, z1, x2, y2, z2, nBins, binSize)

def distanceHistogram3DThreaded(x1, y1, z1, x2, y2, z2, nBins, binSize, split=12):
    """
    distHist.c distanceHistogram3D computed in # split parallel tasks
    """
    hist = np.zeros(nBins)  # the final histogram

    # Compute a series of histograms from points in parallel
    split = np.minimum(split, len(x1))
    split_indices = np.array_split(np.arange(len(x1)),split)
    threads = []
    for _i in np.arange(len(split_indices)):
        _si = split_indices[_i]
        t = threading.Thread(target=distanceHistogram3DThreadWrapper, args=(x1[_si], y1[_si], z1[_si], x2[_si[0]:], y2[_si[0]:], z2[_si[0]:], nBins, binSize, hist))
        threads.append(t)
        t.start()
    
    # Make sure all threads have executed
    for t in threads:
        t.join()

    return hist