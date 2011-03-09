from pylab import *
ioff()

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
visc = 1000  #general solution viscocity - for reactions involving diffusing ligands
visc2 = 1000 #dye-dye effective viscocity - if dyes are tethered, effect of viscocity is likely to be different

#absorption cross-section of fluorescein
abs_xsection = 3.e-16
photons_per_joule_488 = 2.46e18
excitations_per_W_per_cm2_per_us = abs_xsection*photons_per_joule_488/1.e6

#Rate constants for fluorophore-quencher interactions
k10 = 3e1 #kq = (k10 + k11) = 6.6e7 M-1s-1
k11 = 3e1
k12 = 1e1 #appears to be substantially slower than k13
k13 = 1e2  #k13 ~1e8 M-1s-1 (est. from Fig 7)

#permanent photo-bleaching pathways
k14 = 0
k15 = 0

def genReactionScheme(visc, visc2):
    return [
            Reaction('S0 + I <-> S1', k_emmision),      #excitation (this works if we redefine I as being I/I_sat)
            Reaction('S1 <-> S0', k_emmision),          #emmission - we ignore the emitted photon
            Reaction('S1 <-> T1', 6.6e-1),              #intersystem crossing
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
            Reaction('R + q <-> S0 + q', k12/visc),     #should nominally be a reaction with qo, but cannot determine rate constant for this - instead assume that q and qo are in some form of equilibrium and use the relationship to q
            Reaction('X + q <-> S0 + qo', k13/visc),
            Reaction('X <-> P', k14/visc),   #postulated permanent bleaching pathway
            Reaction('R <-> P', k15/visc),   #postulated permanent bleaching pathway
            Reaction('R + O2 <-> S0 + HO2', 5e2/visc),    #oxidation of R by 02 - rate constant from [kaske & linquist `64]
        ]

r = genReactionScheme(visc, visc2)

#Intensity as a fraction of saturation intensity
I0 = 1e-3

#intensity function
#I = Stimulus(1e-3*I0, 1e6*np.array([1, 3,  4, 5,  6, 8,  9, 15, 16, 25, 26, 50, 51, 100]),
#                                   [0, I0, 0, I0, 0, I0, 0, I0, 0,  I0, 0,  I0, 0,  I0])
I = Stimulus(I0, [], [])

#s = System(r,constants={'I':I0, 'q':1e-6, 'O2':0.1*air_sat_O2_conc}, ties={'S1':('I', 'S0')})#
#constants={'q':5e-6, 'O2':0.1*air_sat_O2_conc}
constants={'q':0, 'O2':0}
s = System(r,constants=constants, ties={'S1':('I', 'S0')},stimulae={'I':I})#
s.GenerateGradAndJacCode()
s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M


exPower = I0*k_emmision/excitations_per_W_per_cm2_per_us
print u'Excitation Power: %3.2g W/cm\xB2' % (exPower)
print u'                  = %3.2g mW over a 15 \u03BCm field' % (exPower*(0.0015**2)*1e3)

def plotInitialDecay():
    t = linspace(1, 1e6, 10000)

    res = s.solve(t)

    figure()

    toplot = ['S0', 'T1', 'R', 'X']

    for n in toplot: #res.dtype.names:
        lw = 2
        if n == 'S0':
            lw = 3
        plot((t/1e6), res[n], label=n, lw=lw)

    ylim(0, 1.1*s.initialConditions['S0'])

    legend()
    xlabel('Time [s]')
    ylabel('Concentration [M/L]')


    #detected intensity
    #figure()
    #plot((t/1e6), res['S0']*I(t))

def emod(p, t):
    A, tau = p
    return A*exp(-t/tau)

def stateLifetimes(spec, concs):
    from PYME.Analysis._fithelpers import * 

    figure()

    constants={'q':1e-6, 'O2':0.1*air_sat_O2_conc}
    I = Stimulus(I0, [1e6], [0])

    t = linspace(1, 10e6, 1000)

    tXs = []
    tRs = []

    for i in range(len(concs)):
        constants[spec] = concs[i]

        s = System(r,constants=constants, ties={'S1':('I', 'S0')},stimulae={'I':I})#
        s.GenerateGradAndJacCode()
        s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M

        res = s.solve(t)

        r1 = FitModel(emod, [1e-3, 3], res['X'][t>1e6], t[t>1e6]/1e6 - 1)
        tXs.append(r1[0][1])

        r1 = FitModel(emod, [1e-3, 3], res['R'][t>1e6], t[t>1e6]/1e6 - 1)
        tRs.append(r1[0][1])

    loglog(concs, tXs, label='X', lw=2)
    loglog(concs, tRs, label='R', lw=2)

    ylabel('Time constant [$s^{-1}$]')

    legend()

def dyeConc(concs):
    from PYME.Analysis._fithelpers import *

    figure()

    constants={'q':0, 'O2':0}
    I = Stimulus(I0, [], [])

    t = linspace(1, 1e6, 1000)

    tXs = []
    tRs = []

    s = System(r,constants=constants, ties={'S1':('I', 'S0')},stimulae={'I':I})#
    s.GenerateGradAndJacCode()

    for i in range(len(concs)):  
        s.initialConditions['S0'] = concs[i] #conc of fluorophores on an antibody ~ 100M

        res = s.solve(t)

        r1 = FitModel(emod, [1e-3, 3], res['X'][t>1e6], t[t>1e6]/1e6 - 1)
        tXs.append(r1[0][1])

        r1 = FitModel(emod, [1e-3, 3], res['R'][t>1e6], t[t>1e6]/1e6 - 1)
        tRs.append(r1[0][1])

    loglog(concs, tXs, label='X', lw=2)
    loglog(concs, tRs, label='R', lw=2)

    ylabel('Time constant [$s^{-1}$]')

    legend()

def viscLifetimes(viscs):
    from PYME.Analysis._fithelpers import *

    figure()

    constants={'q':1e-6, 'O2':0.1*air_sat_O2_conc}
    I = Stimulus(I0, [1e6], [0])

    

    tXs = []
    tRs = []

    for i in range(len(viscs)):
        visc = viscs[i]
        visc2 = visc

        if visc <= 10:
            t = linspace(1, 10e6, 10000)
        else:
            t = linspace(1, 10e6, 1000)

        r = genReactionScheme(visc, visc2)

        s = System(r,constants=constants, ties={'S1':('I', 'S0')},stimulae={'I':I})#
        s.GenerateGradAndJacCode()
        s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M

        res = s.solve(t)

        r1 = FitModel(emod, [1e-3, 3], res['X'][t>1e6], t[t>1e6]/1e6 - 1)
        tXs.append(r1[0][1])

        r1 = FitModel(emod, [1e-3, 3], res['R'][t>1e6], t[t>1e6]/1e6 - 1)
        tRs.append(r1[0][1])

    loglog(viscs, tXs, label='X', lw=2)
    loglog(viscs, tRs, label='R', lw=2)

    ylabel('Time constant [$s^{-1}$]')

    legend()


        

def plotMCTraces():
    figure()
    #dm = DiscreteModel(s, ['S0', 'S1', 'T1', 'R', 'X'])
    dm = DiscreteModel(s, ['S0', 'T1', 'R', 'X', 'P'])
    timestep=50.#3e-3
    blinkStartTime = 1 #when to start the blinking simulation
    dm.GenTransitionMatrix(t=[1e6*blinkStartTime], timestep=timestep)


    NSteps = 100000
    t = arange(NSteps)*timestep/1e6 + blinkStartTime

    for i in range(4):
        tr = dm.DoSteps(NSteps)

        subplot(4, 1, i+1)
        step(t[::1], tr[::1] , lw=2)

        yticks(range(5))
        ax = gca()
        ax.set_yticklabels(dm.states)

    xlabel('Time [s]')

def plotConcDep(spec, concs):
    figure()

    constants={'I': I0, 'q':1e-3, 'O2':0.1*air_sat_O2_conc}

    timestep=50.#3e-3
    blinkStartTime = 1 #when to start the blinking simulation
    #dm.GenTransitionMatrix(t=[1e6*blinkStartTime], timestep=timestep)

    NSteps = 100000
    t = arange(NSteps)*timestep/1e6 + blinkStartTime

    for i in range(len(concs)):
        constants[spec] = concs[i]
        
        s = System(r,constants=constants, ties={'S1':('I', 'S0')})#
        s.GenerateGradAndJacCode()
        s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M
        
        dm = DiscreteModel(s, ['S0', 'T1', 'R', 'X', 'P'])
        dm.GenTransitionMatrix(t=[1e6*blinkStartTime], timestep=timestep)

        tr = dm.DoSteps(NSteps)

        subplot(len(concs), 1, i+1)
        step(t[::1], tr[::1] , lw=2)

        yticks(range(5))
        ax = gca()
        ax.set_yticklabels(dm.states)

        ylabel('%3g [M/L]' % concs[i])

    xlabel('Time [s]')

plotInitialDecay()
#plotMCTraces()
#plotConcDep('q', [1e-3, 1e-4, 1e-5, 1e-6])
#title('MEA Concentration')
#plotConcDep('O2', array([1, .1, .01])*air_sat_O2_conc)
#title('O2 Concentration')

#stateLifetimes('q', logspace(-7, -2))
#stateLifetimes('O2', air_sat_O2_conc*logspace(-4, 0))
#viscLifetimes(logspace(0, 3, 10))

draw()

show()
ion()


