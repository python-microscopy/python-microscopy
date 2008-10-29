from pylab import *
from PYME.Analysis._fithelpers import *

a = vstack([cSMI.CDataStack_AsArray(scope.pa.ds, i).ravel() for i in range(8)])

am = a.mean(0)
a_s = a.std(0)

xb = arange(1000, 2000, 10)
yv = zeros(xb.shape)

syv = zeros(xb.shape)

for i in range(len(xb)):
    as_i = a_s[(am >xb[i] - 5)*(am <= xb[i]+5)]
    yv[i] = as_i.mean()
    syv[i] = as_i.std()/sqrt(len(as_i))


def sqrtMod(p, x):
    offset, gain = p
    return sqrt(gain*(x - offset))

I2 = isnan(yv) == False
r = FitModelWeighted(sqrtMod, [1100, 1], yv[I2], syv[I2], xb[I2])
