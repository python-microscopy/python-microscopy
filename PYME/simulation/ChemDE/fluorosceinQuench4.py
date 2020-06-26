# from pylab import *
import matplotlib.pyplot as plt
import numpy as np
plt.ioff()

from reactions import Reaction, System, Stimulus
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

#inverse excited state lifetime
k_emmision = 2.134e2

#the saturating oxygen concentration in water
air_sat_O2_conc = 250e-6

#viscocity - assume reaction rates are proportional to viscocity
#most of our experiments were performed in glycerol, which has a viscocity of ~1200
visc = 1e0  #general solution viscocity - for reactions involving diffusing ligands
visc2 = 1e0 #dye-dye effective viscocity - if dyes are tethered, effect of viscocity is likely to be different

#absorption cross-section of fluorescein
abs_xsection = 3.e-16
photons_per_joule_488 = 2.46e18
excitations_per_W_per_cm2_per_us = abs_xsection*photons_per_joule_488/1.e6

#Rate constants for fluorophore-quencher interactions
k10 = 3e1 #kq = (k10 + k11) = 6.6e7 M-1s-1
k11 = 3e1
k12 = 0*1e-1 #appears to be substantially slower than k13
k13 = 1e2  #k13 ~1e8 M-1s-1 (est. from Fig 7)

#permanent photo-bleaching pathways
k14 = 0e-0
k15 = 0e-0

def genReactionScheme(visc, visc2):
    return [
            #Reaction('S0 + I <-> S1', k_emmision),      #excitation (this works if we redefine I as being I/I_sat)
            #Reaction('S1 <-> S0', k_emmision),          #emmission - we ignore the emitted photon
            Reaction('S0 + I <-> T1', 6.6e-1),              #intersystem crossing
            Reaction('T1 <-> S0', 5e-5),                  #radiationless deactivation
            Reaction('2T1 <-> T1 + S0', 5e2/visc2),       #triplet quenching
            Reaction('T1 + S0 <-> 2S0', 5e1/visc2),         #triplet quenching
            Reaction('2T1 <-> R + X', 6e2/visc2),        #electron transfer
            Reaction('T1 + S0 <-> R + X', 6e2/visc2),         #electron transfer
            Reaction('T1 + X <-> S0 + X', 1e3/visc2, catalysts=['X']),         #quenching by X
            Reaction('T1 + R <-> S0 + R', 1e3/visc2, catalysts=['R']),         #quenching by R
            Reaction('T1 + O2 <-> S0 + O2', 1.56e3/visc),    #physical quenching by 02
            Reaction('T1 + O2 <-> X + HO2', 1.4e2/visc),    #chemical quenching by 02
            Reaction('T1 + q <-> S0 + q', k10/visc),         #quencher
            Reaction('T1 + q <-> R + qo', k11/visc),
            Reaction('R + qo <-> S0 + q', k12/visc),     #should nominally be a reaction with qo, but cannot determine rate constant for this - instead assume that q and qo are in some form of equilibrium and use the relationship to q
            Reaction('X + q <-> S0 + qo', k13/visc),
            #Reaction('X <-> P', k14/visc),   #postulated permanent bleaching pathway
            #Reaction('R <-> P', k15/visc),   #postulated permanent bleaching pathway
            Reaction('R + O2 <-> S0 + HO2', 5e2/visc),    #oxidation of R by 02 - rate constant from [kaske & linquist `64]
            Reaction('qo <-> Cqo', 1e3*.1e-6/visc),     #diffusion of oxidised quencher out of sample
            Reaction('Cqo <-> qo', 1e3*.1e-6/visc),
            Reaction('q <-> Cq', 1e3*.1e-6/visc),
            Reaction('Cq <-> q', 1e3*.1e-6/visc),
            Reaction('O2 <-> Co2', 1e3*.2e-6/visc),
            Reaction('Co2 <-> O2', 1e3*.2e-6/visc),
        ]

r = genReactionScheme(visc, visc2)

#Intensity as a fraction of saturation intensity
I0 = 1e-2

#intensity function
#I = Stimulus(1e-3*I0, 1e6*np.array([1, 3,  4, 5,  6, 8,  9, 15, 16, 25, 26, 50, 51, 100]),
#                                   [0, I0, 0, I0, 0, I0, 0, I0, 0,  I0, 0,  I0, 0,  I0])
I = Stimulus(I0, [], [])

#s = System(r,constants={'I':I0, 'q':1e-6, 'O2':0.1*air_sat_O2_conc}, ties={'S1':('I', 'S0')})#
#constants={'q':10e-3, 'O2':0.1*air_sat_O2_conc}
constants = {'I': 1e-2*1e6*k_emmision, 'Cq':10e-3, 'Cqo':1e-3, 'Co2':1*air_sat_O2_conc}
#constants={'q':10e-3}
s = System(r,constants=constants, ties={'S1':('I', 'S0')})#,stimulae={'I':I})#
s.GenerateGradAndJacCode()
s.initialConditions['S0'] = 1e-2 #conc of fluorophores on an antibody ~ 100M
s.initialConditions['q'] = 10e-3
s.initialConditions['O2'] = 0.1*air_sat_O2_conc


exPower = I0*k_emmision/excitations_per_W_per_cm2_per_us
print(u'Excitation Power: %3.2g W/cm\xB2' % (exPower))
print(u'                  = %3.2g mW over a 15 \u03BCm field' % (exPower*(0.0015**2)*1e3))

def dyedyemodt(p, t):
    '''Model for bleach decay when the rate is dominated by dye-dye interactions.
    The equation (~ 1/t) is a solution to a differential equation of the form:
        dN/dt = -k*N^2
        
    The parameters are:
        A0 - The initial intensity
        k - the rate constant
        b - a constant background
        
    Note that the 1/t curve is shifted to be 1 at t=0.'''
        
    A0, k, c = p
    return A0*k/(t**c + k)

def plotInitialDecay():
    from PYME.Analysis.BleachProfile import rawIntensity
    t = np.linspace(1, 1e5, 10000)

    res = s.solve(t)

    plt.figure()

    toplot = ['S0', 'T1', 'R', 'X', 'O2', 'q']

    for n in toplot: #res.dtype.names:
        lw = 2
        if n == 'S0':
            lw = 3
        plt.plot((t/1e6), res[n], label=n, lw=lw)

    plt.ylim(0, 1.1*s.initialConditions['S0'])

    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration [M/L]')
    
    plt.figure()
    s0 = res['S0']
    t_ = t/1e6
    
    plt.loglog(t_, s0)
    
    fr = rawIntensity.FitModel(dyedyemodt, [s0[0], t_[1] - t_[0], 1.], s0, t_)
    print(fr[0])
    
    plt.loglog(t_, dyedyemodt(fr[0], t_))


import gillespie

G = gillespie.gillespie(s, 1e-21)

plt.ion()




