from pylab import *


from PYME.Analysis.DeClump import deClump


t = arange(0, 200, .2)
print len(t)
x = randn(1000)
y = randn(1000)
delta_x = .5*ones(x.shape)

asg = deClump.findClumps(t.astype('i'), x.astype('f4'), y.astype('f4'), delta_x.astype('f4'))

print asg