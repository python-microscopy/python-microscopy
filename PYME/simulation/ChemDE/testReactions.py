from pylab import *
ioff()

import reactions

s = reactions.System([
    reactions.Reaction('2A + B <-> C', .1),
    reactions.Reaction('2C + B <-> D', .1)])

s.GenerateGradAndJacCode()
    
s.initialConditions['A'] = 1
s.initialConditions['B'] = 1

t = linspace(.1, 10, 100)

res = s.solve(t)

for n in res.dtype.names:
    plot(t, res[n], label=n)

legend()
show()

