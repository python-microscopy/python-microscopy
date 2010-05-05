from pylab import *


from PYME.Analysis.DeClump import deClump


t = arange(0, 200, .02)
print len(t)
x = randn(10000)
y = randn(10000)
delta_x = .05*ones(x.shape)

asg = deClump.findClumps(t.astype('i'), x.astype('f4'), y.astype('f4'), delta_x.astype('f4'), 2)

print asg