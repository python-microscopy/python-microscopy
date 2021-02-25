# from pylab import *
import matplotlib.pyplot as plt
import numpy as np
#ioff()

import reactions
from discreteReactions import DiscreteModel

r = [
    reactions.Reaction('Off + IA <-> On + IA', .0005),
    reactions.Reaction('On + I <-> Dark + I', .1),
    reactions.Reaction('On + I <-> Off + I', .1),
    reactions.Reaction('Dark <-> On', .1),
    reactions.Reaction('On + I <-> Bleached', .1),
    
]

s = reactions.System(r, constants={'I':1, 'IA' : 1})#
s.GenerateGradAndJacCode()
s.initialConditions['Off'] = 1

#print s.getDEs()

t = np.linspace(.1, 1e3, 300)

res = s.solve(t)

for n in res.dtype.names:
    plt.plot(t, res[n], label=n)

plt.legend()
plt.show()

plt.figure()
#dm = DiscreteModel(s, ['S0', 'S1', 'T1', 'R', 'X'])
dm = DiscreteModel(s, ['Off', 'On', 'Dark', 'Bleached'])
timestep=2.#3e-3
blinkStartTime = 0.1 #when to start the blinking simulation
dm.GenTransitionMatrix(t=[blinkStartTime], timestep=timestep)


NSteps = 10000
t = np.arange(NSteps)*timestep + blinkStartTime

for i in range(4):
    tr = dm.DoSteps(NSteps)

    plt.subplot(4, 1, i+1)
    plt.step(t[::1], tr[::1] , lw=2)

    plt.yticks(range(5))
    ax = plt.gca()
    ax.set_yticklabels(dm.states)

plt.xlabel('Time [s]')

#def countEvents(trace, threshold):
#    nEvents = 0
#    lastObs = -1e9
#
#    for i in range(len(trace)):
#        if trace[i]:
#            if (i- lastObs) >= threshold:
#                nEvents += 1
#            lastObs = i
#    return nEvents

from countEvents import countEvents

densities = [1,2,5,10,20,50,100] #molecules/diffraction limited volume
#densities = [100]

eventCounts = {}
trange = np.hstack([np.arange(1,100), np.logspace(2, 4, 100)])

for d in densities:
    eventCounts[d] = np.zeros(trange.shape)

nIters = 10
NSteps = 100000
for j in np.arange(nIters):

    traces = []

    for i in np.arange(100):
        traces.append(dm.DoSteps(NSteps))

    for d in densities:
        for k in range(100/d):
            tr = ((np.vstack(traces[(k*d):(k*d + d)]) == 1).sum(0) > .5).astype('int32')
            eventCounts[d] += np.array([countEvents(tr, td) for td in trange])

plt.figure()
for d in densities:
    plt.semilogx(trange, eventCounts[d]/(100*nIters), label='%d /diff. limited vol.' % d)

plt.legend()






