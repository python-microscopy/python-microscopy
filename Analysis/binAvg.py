import numpy as np
from pylab import *

def binAvg(binVar, indepVar, bins):
    bm = []
    bs = []
    bn = []

    for el, er in zip(bins[:-1], bins[1:]):
        v = indepVar[(binVar >= el)*(binVar < er)]

        bn.append(len(v))
        bm.append(v.mean())
        bs.append(v.std())

    return np.array(bn), np.array(bm), np.array(bs)


def errorPlot(filter, bins):
    x = (bins[:-1] + bins[1:])/2.

    a1 = axes()

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_x0'], bins)

    a1.plot(x, bm, lw=2, c='b', label='x')
    a1.fill_between(x, maximum(bm-bs, 0), bm + bs, facecolor='b', alpha=0.2)

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_y0'], bins)

    a1.plot(x, bm, lw=2, c='g', label='y')
    a1.fill_between(x, maximum(bm-bs, 0), bm + bs, facecolor='g', alpha=0.2)

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_z0'], bins)

    a1.plot(x, bm, lw=2, c='r', label='z')
    a1.fill_between(x, maximum(bm-bs, 0), bm + bs, facecolor='r', alpha=0.2)

    ylabel('Fit Error [nm]')
    xlabel('Defocus [nm]')
    legend()
    ylim(0, 120)


    a2 = twinx()

    a2.plot(x, bn, 'k')
    ylabel('Number of fitted events')