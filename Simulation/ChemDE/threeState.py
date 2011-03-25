from pylab import *
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

t = linspace(.1, 1e3, 300)

res = s.solve(t)

for n in res.dtype.names:
    plot(t, res[n], label=n)

legend()
show()

figure()
#dm = DiscreteModel(s, ['S0', 'S1', 'T1', 'R', 'X'])
dm = DiscreteModel(s, ['Off', 'On', 'Dark', 'Bleached'])
timestep=2.#3e-3
blinkStartTime = 0.1 #when to start the blinking simulation
dm.GenTransitionMatrix(t=[blinkStartTime], timestep=timestep)


NSteps = 10000
t = arange(NSteps)*timestep + blinkStartTime

for i in range(4):
    tr = dm.DoSteps(NSteps)

    subplot(4, 1, i+1)
    step(t[::1], tr[::1] , lw=2)

    yticks(range(5))
    ax = gca()
    ax.set_yticklabels(dm.states)

xlabel('Time [s]')

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

eventCounts = {}
for d in densities:
    eventCounts[d] = zeros(300)

trange = logspace(0, 4, 300)

nIters = 10
NSteps = 100000
for j in xrange(nIters):

    traces = []

    for i in xrange(100):
        traces.append(dm.DoSteps(NSteps))

    for d in densities:
        for k in range(100/d):
            tr = ((vstack(traces[(k*d):(k*d + d)]) == 1).sum(0) > .5).astype('int32')
            eventCounts[d] += array([countEvents(tr, td) for td in trange])

figure()
for d in densities:
    semilogx(trange, eventCounts[d]/(100*nIters), label='%d /diff. limited vol.' % d)

legend()






