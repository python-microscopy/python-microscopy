# from pylab import *
import matplotlib.pyplot as plt
import numpy as np
#ioff()

import reactions

#Fl = fluorophore
#Flr = radical fluorophore
#Th = thiol
#FlThA, FlrThA = thiol-flurophore pre-ascociated complex
#FlThB = thiol bound flourophore
#FlO = oxidised (bleached) fluorophore
#I = light intensity (note conservation)

r = [
    reactions.Reaction('Fl + I <-> Flr + I', k_forward=.1, k_reverse=0),
    reactions.Reaction('Flr <-> Fl', k_forward=.1, k_reverse=0),
    reactions.Reaction('Fl + Th <-> FlThA', k_forward=.1, k_reverse=.1),
    reactions.Reaction('Flr + Th <-> FlrThA', k_forward=.1, k_reverse=.1),
    reactions.Reaction('FlThA + I <-> FlrThA + I', k_forward=.1, k_reverse=0),
    reactions.Reaction('FlrThA <-> FlThA ', k_forward=.1, k_reverse=0),
    reactions.Reaction('FlrThA <-> FlrThB', k_forward=1, k_reverse=.01),
    reactions.Reaction('Flr + O <-> FlO', k_forward=.01, k_reverse=0),
]

#intensity function
def I(t):
    return 1*(t > 50)

#s = reactions.System(r,stimulae={'I':reactions.Stimulus(1, [], [])}) #, constants={'I':1})#
s = reactions.System(r, constants={'I':1})#
s.GenerateGradAndJacCode()
s.initialConditions['Fl'] = 1
s.initialConditions['Th'] = 1
#s.initialConditions['O'] = .5
#s.initialConditions['I'] = 1

#print s.getDEs()

t = np.linspace(.1, 300, 300)

res = s.solve(t)

for n in res.dtype.names:
    plt.plot(t, res[n], label=n)

plt.legend()
plt.show()

