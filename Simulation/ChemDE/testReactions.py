from pylab import *
ioff()

import reactions

r = reactions.Reaction('2A + B <-> C', k_reverse=.01)
r2 = reactions.Reaction('2C + B <-> D', k_forward=.1, k_reverse=.01)

s = reactions.System([r, r2])
s.initialConditions['A'] = 1
s.initialConditions['B'] = 1

t = linspace(.1, 10, 100)

res = s.solve(t)

for n in res.dtype.names:
    plot(t, res[n], label=n)

legend()
show()

