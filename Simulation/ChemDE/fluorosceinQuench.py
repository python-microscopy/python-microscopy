from pylab import *
#ioff()

from reactions import Reaction, System
from discreteReactions import DiscreteModel

#s   - singlet (nb lower case to keep sympy happy)
#Se  - excited singlet
#Te  - excited triplet
#R   - semi-reduced dye
#X   - semi oxidised dye
#Q   - quencher
#Qo  - oxidised quencher
#O2  - oxygen (O2)
#H02 - superoxide (H02)
#I  - light (as a fraction of Isat, the saturation intensity)

#using constants for fluorescein from:
#Influence of the triplet excited state on the photobleaching kinetics of fluorescein in microscopy. Song et Al, Biophys J. 1996
#time base in us (ie constants are divided by 1e6)
k_emmision = 2.134e2

air_sat_O2_conc = 250e-6

visc = 1000
visc2 = 1000 #assume fluorophore-fluorophore interactions are mostly fret based and not reliant on diffusion

abs_xsection = 3.e-16
photons_per_joule_488 = 2.46e18
excitations_per_W_per_cm2_per_us = abs_xsection*photons_per_joule_488/1.e6

k10 = 3e1 #kq = (k10 + k11) = 6.6e7 M-1s-1
k11 = 3e1
k12 = 1e1 #appears to be substantially slower than k13
k13 = 1e2  #k13 ~1e8 M-1s-1 (est. from Fig 7)

k14 = 1e2

r = [
    Reaction('S0 + I <-> S1', k_emmision),   #excitation
    Reaction('S1 <-> S0', k_emmision),          #emmission - we ignore the emitted photon
    Reaction('S1 <-> T1', 6.6e-1),              #intersystem crossing
    Reaction('T1 <-> S0', 5e-5),                  #radiationless deactivation
    Reaction('2T1 <-> T1 + S0', 5e2/visc2),       #triplet quenching
    Reaction('T1 + S0 <-> 2S0', 5e1/visc2),         #triplet quenching
    Reaction('2T1 <-> R + X', 6e2/visc2),        #electron transfer
    Reaction('T1 + S0 <-> R + X', 6e2/visc2),         #electron transfer
    Reaction('T1 + X <-> S0 + X', 1e3/visc2),         #quenching by X
    Reaction('T1 + R <-> S0 + R', 1e3/visc2),         #quenching by R
    Reaction('T1 + O2 <-> S0 + O2', 1.56e3/visc),    #physical quenching by 02
    Reaction('T1 + O2 <-> X + HO2', 1.4e2/visc),    #chemical quenching by 02
    Reaction('T1 + q <-> S0 + q', k10/visc),         #quencher
    Reaction('T1 + q <-> R + qo', k11/visc),
    Reaction('R + qo <-> S0 + q', k12/visc),
    Reaction('X + q <-> S0 + qo', k13/visc),
    Reaction('X + O2 <-> XO', k14/visc),   #postulated permanent bleaching pathway
]

I0 = 1e-4

#intensity function
def I(t):
    return I0*(t > 50)

#s = System(r,stimulae={'I':I}) #, constants={'I':1})#
s = System(r,constants={'I':I0, 'q':1e-6, 'O2':0.01*air_sat_O2_conc}, ties={'S1':(I0, 'S0')})#
s.GenerateGradAndJacCode()
s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M
#s.initialConditions['S1'] = s.initialConditions['S0']*I0 #this equilibrium will be reached really fast - help the solver out
#s.initialConditions['q'] = 1e-5
#s.initialConditions['O2'] = 0.1*air_sat_O2_conc
#s.initialConditions['I'] = 1


exPower = I0*k_emmision/excitations_per_W_per_cm2_per_us
print u'Excitation Power: %3.2g W/cm\xB2' % (exPower)
print u'                  = %3.2g mW over a 15 \u03BCm field' % (exPower*(0.0015**2)*1e3)

#print s.getDEs()

t = linspace(1, 1e6, 10000)

res = s.solve(t)

figure()

toplot = ['S0', 'T1', 'R', 'X', 'XO']

for n in toplot: #res.dtype.names:
    plot((t/1e6), res[n], label=n, lw=2)

ylim(0, 1.1*s.initialConditions['S0'])

legend()


figure()
#dm = DiscreteModel(s, ['S0', 'S1', 'T1', 'R', 'X'])
dm = DiscreteModel(s, ['S0', 'T1', 'R', 'X', 'XO'])
timestep=10.#3e-3
dm.GenTransitionMatrix(t=[1e5], timestep=timestep)

for i in range(4):
    tr = dm.DoSteps(100000)

    subplot(4, 1, i+1)
    step(arange(len(tr))*timestep/1e6, tr , lw=2)
    
    yticks(range(5))
    ax = gca()
    ax.set_yticklabels(dm.states)

draw()

show()


