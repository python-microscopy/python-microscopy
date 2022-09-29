"""
A simple utility function to look for (and keep) pairs in 2 sets of localisations.

Useful for comparing different analysis and/or detection modes on the same data. 
Could also be used on, e.g. beads in different colour channels.

David Baddeley 2022
"""

import numpy as np

def _idx(t):
    idx = np.concatenate([[0,], np.argwhere(np.diff(t) > 0).squeeze() + 1])
    idr_r = {t[i]:i for i in idx}
    idx_e = {t[i]:i+1 for i in idx[1:]-1}
    idx_e [t[-1]] = len(t) - 1

    return idx, idr_r, idx_e

def keep_pairs(x0, y0, t0, x1, y1, t1):
    """
    Find all localisations in (x0,y0,t0) which have a corresponding localisation in (x1,y1,t1) and return
    the matching pairs. Uses a simple distance metric to find pairs within a frame.

    TODO:
    - Return indices so that additional columns etc ... can be examined
    - Potentially change to finding nearest match **without** replacement. As coded it is theoretically possible
      for 2 localisations in set 0 to map to a single localisation in set 1 (and vice versa), although this is
      expected to be rare in practive.
    """
    idx0, f0, e0, = _idx(t0)
    idx1, f1, e1 = _idx(t1)

    o0 = []
    o1 = []
    do = []

    idxs = np.array(list(f0.keys()), dtype='i')
    for i in idxs:
        if i in f1:
            #skip frames which are completely missing from t1

            x0_i, y0_i, t0_i = x0[f0[i]:e0[i]], y0[f0[i]:e0[i]], t0[f0[i]:e0[i]]
            x1_i, y1_i, t1_i = x1[f1[i]:e1[i]], y1[f1[i]:e1[i]], t1[f1[i]:e1[i]]

            d = (x0_i[:,None] - x1_i[None,:])**2 + (y0_i[:,None] - y1_i[None,:])**2

            if len(x0_i) > len(x1_i):
                mx = np.atleast_1d(d.argmin(0).squeeze())
                dm = d[mx, np.arange(len(mx), dtype='i')].squeeze()
                o1.append(np.vstack([x1_i, y1_i, t1_i]))
                o0.append(np.vstack([x0_i[mx], y0_i[mx], t0_i[mx]]))
                do.append(dm)
            else:
                mx = np.atleast_1d(d.argmin(1).squeeze())
                dm = d[np.arange(len(mx), dtype='i'),mx].squeeze()
                o1.append(np.vstack([x1_i[mx], y1_i[mx], t1_i[mx]]))
                o0.append(np.vstack([x0_i, y0_i, t0_i]))
                do.append(dm)

            #if i < 50:
            #    print(d.shape, dm.shape, mx.shape)
            #    print(dm)

    #print(o0[0].shape, len(o0), o0[1].shape, do[0].shape)

    do = np.hstack(do)
    return np.concatenate(o0, 1), np.concatenate(o1, 1), do

