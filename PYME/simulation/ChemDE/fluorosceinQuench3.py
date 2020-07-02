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
visc = 1e3  #general solution viscocity - for reactions involving diffusing ligands
visc2 = 1e3 #dye-dye effective viscocity - if dyes are tethered, effect of viscocity is likely to be different

#absorption cross-section of fluorescein
abs_xsection = 3.e-16
photons_per_joule_488 = 2.46e18
excitations_per_W_per_cm2_per_us = abs_xsection*photons_per_joule_488/1.e6

#Rate constants for fluorophore-quencher interactions
k10 = 3e1 #kq = (k10 + k11) = 6.6e7 M-1s-1
k11 = 3e1
k12 = 1e-1 #appears to be substantially slower than k13
k13 = 1e2  #k13 ~1e8 M-1s-1 (est. from Fig 7)

#permanent photo-bleaching pathways
k14 = 0e-0
k15 = 0e-0

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
            Reaction('R + qo <-> S0 + q', k12/visc),     #should nominally be a reaction with qo, but cannot determine rate constant for this - instead assume that q and qo are in some form of equilibrium and use the relationship to q
            Reaction('X + q <-> S0 + qo', k13/visc),
            Reaction('X <-> P', k14/visc),   #postulated permanent bleaching pathway
            Reaction('R <-> P', k15/visc),   #postulated permanent bleaching pathway
            Reaction('R + O2 <-> S0 + HO2', 5e2/visc),    #oxidation of R by 02 - rate constant from [kaske & linquist `64]
        ]

r = genReactionScheme(visc, visc2)

#Intensity as a fraction of saturation intensity
I0 = 1e-4

#intensity function
#I = Stimulus(1e-3*I0, 1e6*np.array([1, 3,  4, 5,  6, 8,  9, 15, 16, 25, 26, 50, 51, 100]),
#                                   [0, I0, 0, I0, 0, I0, 0, I0, 0,  I0, 0,  I0, 0,  I0])
I = Stimulus(I0, [], [])

#s = System(r,constants={'I':I0, 'q':1e-6, 'O2':0.1*air_sat_O2_conc}, ties={'S1':('I', 'S0')})#
#constants={'q':10e-3, 'O2':0.1*air_sat_O2_conc}
constants = {}
#constants={'q':10e-3}
s = System(r,constants=constants, ties={'S1':('I', 'S0')},stimulae={'I':I})#
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
        plot((t/1e6), res[n], label=n, lw=lw)

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


    #detected intensity
    #figure()
    #plot((t/1e6), res['S0']*I(t))

def emod(p, t):
    A, tau = p
    return A*np.exp(-t/tau)

def stateLifetimes(spec, concs):
    from PYME.Analysis._fithelpers import FitModel

    plt.figure()

    constants={'q':10e-3, 'O2':0.1*air_sat_O2_conc}
    I = Stimulus(0, [], [])

    t = np.linspace(1, 10e6, 1000)

    tXs = []
    tRs = []

    nConc = constants[spec]

    for i in range(len(concs)):
        constants[spec] = concs[i]

        s = System(r,constants=constants, ties={'S1':('I', 'S0')},stimulae={'I':I})#
        s.GenerateGradAndJacCode()
        #s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M
        s.initialConditions['X'] = 1e-3
        s.initialConditions['R'] = 1e-3

        res = s.solve(t)

        r1 = FitModel(emod, [1e-3, 3], res['X'], t/1e6)
        tXs.append(r1[0][1])

        r1 = FitModel(emod, [1e-3, 3], res['R'], t/1e6)
        tRs.append(r1[0][1])

        print(max(tXs[-1], tRs[-1]))
        #t = linspace(1, 5e6*max(max(tXs[-1], tRs[-1]), 1e-3), 1000)

    plt.loglog(concs, tXs, label='X', lw=2)
    plt.loglog(concs, tRs, label='R', lw=2)

    plt.plot([nConc, nConc], plt.ylim(), 'k--')

    plt.ylabel('Time constant [s]')
    plt.xlabel('[%s]' % spec)

    plt.legend()

def stateLifetimes2(spec, concs, r, labelAddition = '',  constants={'q':10e-3, 'O2':0.1*air_sat_O2_conc}, **kwargs):
    lineSpec = {'lw':2}
    lineSpec.update(kwargs)

    #figure()

    #constants={'q':10e-3, 'O2':0.1*air_sat_O2_conc}
    I = Stimulus(0, [], [])

    t = np.linspace(1, 1e5, 1000)

    tXs = []
    tRs = []

    nConc = constants[spec]

    rates = []

    initConc = 1e-3

    s = System(r,constants={}, ties={'S1':('0', 'S0')},stimulae={'I':I})#
    s.GenerateGradAndJacCode()
    s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M
    #s.initialConditions['X'] = initConc
    #s.initialConditions['R'] = initConc

    s.initialConditions['q'] = constants['q']
    s.initialConditions['O2'] = constants['O2']

    for i in range(len(concs)):
        s.initialConditions[spec] = concs[i]
        rates.append(s.GradFcn(0, s.initialConditions.view('f8')).view(s.dtype))

    rates = np.hstack(rates)

    plt.loglog(concs, 1e-6/(-rates['X']/initConc), c = 'b', label='X' + labelAddition, **lineSpec)
    plt.loglog(concs, 1e-6/(-rates['R']/initConc), c = 'g', label='R' + labelAddition, **lineSpec)

    plt.plot([nConc, nConc], ylim(), 'k--')

    plt.ylabel('Dark state lifetime [s]')
    plt.xlabel('[%s]' % spec)

    plt.legend()
    
def stateLifetimes3(spec, concs, r, labelAddition = '',  constants={'q':10e-3, 'O2':0.1*air_sat_O2_conc}, **kwargs):
    lineSpec = {'lw':2}
    lineSpec.update(kwargs)

    #figure()

    #constants={'q':10e-3, 'O2':0.1*air_sat_O2_conc}
    I = Stimulus(I0, [9e4], [0])

    t = np.linspace(1, 1e5, 10000)

    tXs = []
    tRs = []

    nConc = constants[spec]

    rates = []

    initConc = 1e-3

    s = System(r,constants={}, ties={'S1':('I', 'S0')},stimulae={'I':I})#
    s.GenerateGradAndJacCode()
    s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M
    #s.initialConditions['X'] = initConc
    #s.initialConditions['R'] = initConc

    s.initialConditions['q'] = constants['q']
    s.initialConditions['O2'] = constants['O2']

    for i in range(len(concs)):
        s.initialConditions[spec] = concs[i]
        res, r = s.solve(t, True)
        #print res[-1].dtype
        #print res[-1].view('9f8')
        rates.append(r[-1])

    rates = np.hstack(rates)
    
    #plot(concs, 1e-6/(-rates['X']/initConc))

    plt.loglog(concs, 1e-6/(-rates['X']/initConc), c = 'b', label='X' + labelAddition, **lineSpec)
    plt.loglog(concs, 1e-6/(-rates['R']/initConc), c = 'g', label='R' + labelAddition, **lineSpec)

    plt.plot([nConc, nConc], plt.ylim(), 'k--')

    plt.ylabel('Dark state lifetime [s]')
    plt.xlabel('[%s]' % spec)

    plt.legend()

def dyeConc2(concs):
    constants={'q':5e-3, 'O2':0.1*air_sat_O2_conc}
    #constants = {}
    #constants={'q':0, 'O2':0}
    s = System(r,constants=constants, ties={'S1':('I', 'S0')},stimulae={'I':I})#
    s.GenerateGradAndJacCode()

    rates = []
    for c in concs:
        s.initialConditions['S0'] = c

        res = s.solve([1e3])

        rates.append(s.GradFcn(0, res.view('f8')).view(s.dtype))

    rates = np.hstack(rates)

    plt.figure()
    #plot(concs, rates['S0']/concs)
    plt.plot(concs, rates['T1']/concs)

def dyeConc(concs):
    from PYME.Analysis._fithelpers import FitModel

    plt.figure()

    constants={'q':0, 'O2':0}
    I = Stimulus(I0, [], [])

    t = np.linspace(1, 1e6, 1000)

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

    plt.loglog(concs, tXs, label='X', lw=2)
    plt.loglog(concs, tRs, label='R', lw=2)

    plt.ylabel('Time constant [$s^{-1}$]')

    plt.legend()

def viscLifetimes(viscs):
    from PYME.Analysis._fithelpers import *

    plt.figure()

    constants={'q':1e-6, 'O2':0.1*air_sat_O2_conc}
    I = Stimulus(I0, [1e6], [0])

    

    tXs = []
    tRs = []

    for i in range(len(viscs)):
        visc = viscs[i]
        visc2 = visc

        if visc <= 10:
            t = np.linspace(1, 10e6, 10000)
        else:
            t = np.linspace(1, 10e6, 1000)

        r = genReactionScheme(visc, visc2)

        s = System(r,constants=constants, ties={'S1':('I', 'S0')},stimulae={'I':I})#
        s.GenerateGradAndJacCode()
        s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M

        res = s.solve(t)

        r1 = FitModel(emod, [1e-3, 3], res['X'][t>1e6], t[t>1e6]/1e6 - 1)
        tXs.append(r1[0][1])

        r1 = FitModel(emod, [1e-3, 3], res['R'][t>1e6], t[t>1e6]/1e6 - 1)
        tRs.append(r1[0][1])

    plt.loglog(viscs, tXs, label='X', lw=2)
    plt.loglog(viscs, tRs, label='R', lw=2)

    plt.ylabel('Time constant [$s^{-1}$]')

    plt.legend()


        

def plotMCTraces():
    import matplotlib.patches
    plt.figure(figsize=(4.5, 1.5))
    #dm = DiscreteModel(s, ['S0', 'S1', 'T1', 'R', 'X'])
    dm = DiscreteModel(s, ['S0', 'T1', 'R', 'X', 'P'])
    timestep=10#3e-3
    blinkStartTime = 1 #when to start the blinking simulation
    dm.GenTransitionMatrix(t=[1e6*blinkStartTime], timestep=timestep)


    NSteps = 5000000
    t = np.arange(NSteps)*timestep/1e6 + blinkStartTime

    for i in range(1):
        tr = dm.DoSteps(NSteps)

        #subplot(1, 1, i+1)
        plt.axes([.15,.3,.8,.65])
        ax = plt.gca()

        p = matplotlib.patches.Rectangle([0,.5], .5,3,facecolor=[.8,.8,.8], edgecolor=None)
        ax.add_patch(p)

        p = matplotlib.patches.Rectangle([0,-.5], .5,1,facecolor=[.9,1.,.8], edgecolor=None)
        ax.add_patch(p)

        plt.step((t[::1] - 1)*1e3, tr[::1])# , lw=1)

        plt.yticks(range(4))
        #ax = gca()
        #ax.set_yticklabels(dm.states)

        ax.set_yticklabels(['S0/S1', 'T1', '$D^*_{red}$', '$D^*_{ox}$'])

        plt.axis([0, t.max(), 3.5, -.5])

    #ylim(-.2, 4.1)

    plt.xlabel(u'Time [ms]')

def plotConcDep(spec, concs):
    plt.figure()

    constants={'I': I0, 'q':1e-3, 'O2':0.1*air_sat_O2_conc}

    timestep=50.#3e-3
    blinkStartTime = 1 #when to start the blinking simulation
    #dm.GenTransitionMatrix(t=[1e6*blinkStartTime], timestep=timestep)

    NSteps = 100000
    t = np.arange(NSteps)*timestep/1e6 + blinkStartTime

    for i in range(len(concs)):
        constants[spec] = concs[i]
        
        s = System(r,constants=constants, ties={'S1':('I', 'S0')})#
        s.GenerateGradAndJacCode()
        s.initialConditions['S0'] = 1e-3 #conc of fluorophores on an antibody ~ 100M
        
        dm = DiscreteModel(s, ['S0', 'T1', 'R', 'X', 'P'])
        dm.GenTransitionMatrix(t=[1e6*blinkStartTime], timestep=timestep)

        tr = dm.DoSteps(NSteps)

        plt.subplot(len(concs), 1, i+1)
        plt.step(t[::1], tr[::1] , lw=2)

        plt.yticks(range(5))
        ax = plt.gca()
        ax.set_yticklabels(dm.states)

        plt.ylabel('%3g [M/L]' % concs[i])

    plt.xlabel('Time [s]')

#plotInitialDecay()
#plotMCTraces()
#dyeConc2(logspace(-6, 2))
#plotConcDep('q', [1e-3, 1e-4, 1e-5, 1e-6])
#title('MEA Concentration')
#plotConcDep('O2', array([1, .1, .01])*air_sat_O2_conc)
#title('O2 Concentration')

#stateLifetimes('q', logspace(-9, -2))
#figure()
#r = genReactionScheme(1e3, 1e3)
#stateLifetimes3('q', logspace(-9, -1, 20), r, '  $\eta = 1$, [$O_2$] = 10%')
#r = genReactionScheme(1000, 1000)
#stateLifetimes2('q', logspace(-9, -1), r, '  $\eta = 1000$, [$O_2$] = 10%', ls='--', lw=1)

#figure()
#r = genReactionScheme(1, 1)
#stateLifetimes3('O2', air_sat_O2_conc*logspace(-1, 0, 20), r, '  $\eta = 1$, [MEA] = 10mM')
#stateLifetimes2('O2', air_sat_O2_conc*logspace(-9, 0), r, '  $\eta = 1$, [MEA] = 1nM', constants={'q':1e-9, 'O2':0.1*air_sat_O2_conc}, ls=':', lw=1)
#r = genReactionScheme(1000, 1000)
#stateLifetimes2('O2', air_sat_O2_conc*logspace(-9, 0), r, '  $\eta = 1000$, [MEA] = 10mM', ls='--', lw=1)
#viscLifetimes(logspace(0, 3, 10))

#draw()

import networkx as nx

plt.figure()

G = nx.DiGraph()

fluor_states = ['S0', 'S1', 'T1', 'R', 'X', 'P']

for n, r in enumerate(s.reactions):
    for reag in r.reagents:
        for prod in r.products:
            if reag in fluor_states and prod in fluor_states:
                G.add_edge(reag, prod)
            #G.add_edge(reag, n)
        
    
        
#nx.draw(G)
    

#show()
#ion()


import cherrypy

class WebView(object):
    def __init__(self, system):
        self.system = system
        self.system.fluor_states = fluor_states
        
    @cherrypy.expose
    def index(self, templateName='ChemDE.html'):
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader('.'))
        
        #if self.dsviewer.image.data.shape[2] > 1:
            #3D image
       #     templateName=
        
        template = env.get_template(templateName)
        
        return template.render(system=self.system)
    
    @cherrypy.expose
    def ensembleDecay(self, **kwargs):
        cherrypy.response.headers['Content-Type'] = "application/json"
        import gviz_api
        t = np.linspace(1, 1e5, 1000)
        
        for sp in self.system.species:
            if sp in kwargs.keys():
                self.system.initialConditions[sp] = float(kwargs[sp])
    
        res = self.system.solve(t)
        
        print(kwargs.keys())
        
        table = gviz_api.DataTable([('t', 'number')] + [(n, 'number') for n in res.dtype.names])
        table.LoadData([[ti*1e-6,] + list(r) for ti, r in zip(t, res)])
        
        tqx = {}
        for var in kwargs['tqx'].split(';'):
            n, v = var.split(':')
            tqx[n]=v
    
        return table.ToJSonResponse(req_id=int(tqx['reqId']))
        
    @cherrypy.expose
    def blinking(self, **kwargs):
        cherrypy.response.headers['Content-Type'] = "application/json"
        import gviz_api

        
        for sp in self.system.species:
            if sp in kwargs.keys():
                self.system.initialConditions[sp] = float(kwargs[sp])
        
        dm = DiscreteModel(self.system, ['S0', 'T1', 'R', 'X', 'P'])
        timestep=10#3e-3
        blinkStartTime = 1 #when to start the blinking simulation
        dm.GenTransitionMatrix(t=[1e6*blinkStartTime], timestep=timestep)


        NSteps = 500000
        t = np.arange(NSteps)*timestep/1e6 + blinkStartTime
        
        tr = np.zeros((NSteps, 1), 'i')

        for i in range(tr.shape[1]):
            tr[:, i] = dm.DoSteps(NSteps)
        
        m = np.ones(len(t), 'i')
        dtr = (np.diff(tr, axis=0) != 0).sum(axis=1)
        m[1:] = dtr
        m[:-1] += dtr
        
        m[0] = 1
        m[-1] = 1
        
        m = m > .5
        
        #tr += 4*np.arange(4)[None, :]
        
        #print m.shape, tr[m, :].shape
        
        print(kwargs.keys())
        
        table = gviz_api.DataTable([('t', 'number')] + [(n, 'number') for n in ['Mol%d'%d for d in range(tr.shape[1])]])
        table.LoadData([[ti,] + [int(ri) for ri in r] for ti, r in zip(t[m], tr[m, :])])
        
        tqx = {}
        for var in kwargs['tqx'].split(';'):
            n, v = var.split(':')
            tqx[n]=v
    
        return table.ToJSonResponse(req_id=int(tqx['reqId']))

        

cherrypy.quickstart(WebView(s))
