import numpy as np
from StateMC import StateMC

class DiscreteModel:
    def __init__(self, system, states):
        self.system = system
        #self.var = discreteVar
        self.states = states
        self.nStates = len(states)

        self.state = 0

        self.hist = []

    def _getConc(self, expr, concs, t):
        for n in concs.dtype.names:
            locals()[n] = concs[n]

        locals().update(self.system.constants)

        for key, fcn in self.system.stimulae.items():
            locals()[key] = fcn(t)

        return eval(expr)

    def _resolveTies(self, state, concs, t):
        if state in self.system.ties.keys():
            sc, state_ = self.system.ties[state]
            
            return self._getConc(sc, concs, t), state_
        else:
            return 1, state


    def GenTransitionMatrix(self, t, concs = None, timestep=1):
        self.M = np.zeros((len(self.states), len(self.states)))

        tied = self.system.ties.keys()
        #print tied

        sStates = set(self.states + tied)

        if concs == None:
            concs = self.system.solve(t)

        for r in self.system.reactions:
            #print '\n' + r.reaction_equation
            reags = r.reagents.keys()
            prods = r.products.keys()
            ReagStates = sStates.intersection(reags).difference(r.catalysts)
            ProdStates = sStates.intersection(prods).difference(r.catalysts)

            for state in ReagStates:
                scale, state_ = self._resolveTies(state, concs, t)

                n = self.states.index(state_)

                p = r.k_forward
                for st in reags:
                    sc, st_ = self._resolveTies(st, concs, t)
                    mol = r.reagents[st]
                    if st_ == state_: #we are in this state - act as if concentration is one
                        mol -= 1

                    #print state_, st_, mol

                    if st in self.system.constants.keys():
                        p*= self.system.constants[st]**mol
                    else:
                        #p*= sc*concs[st_]**mol
                        p *= sc*self._getConc(st_, concs, t)**mol

                    #print state_, st_, mol, p
                    
#                for st in reags:
#                    mol = r.reagents[st]
#                    if st == state: #we are in this state - act as if concentration is one
#                        mol -= 1
#
#                    p*= concs[st]**mol

                #equally likely to go into either of the product states
                p = p/sum([r.products[st] for st in ProdStates])

                for st in ProdStates:
                    sc, st_ = self._resolveTies(st, concs, t)

                    if not st_ == state_:

                        m = self.states.index(st_)

                        self.M[n, m] += r.products[st]*p

        self.M *= timestep
        self.MC = np.cumsum(self.M, 1)
        
        if self.MC.max() >=1:
            raise RuntimeError('Error: Sum of probabilities > 1 - try a shorter timestep')


        
    def DoStep(self):
        r = np.random.rand()
        newState = self.MC[self.state, :].searchsorted(r)

        if newState < self.nStates:
            self.state = newState

        self.hist.append(self.state)

    def DoSteps(self, N, startState = 0):
        #self.state = startState

        #trace = np.zeros(N)
        #for i in range(N):
        #    r = np.random.rand()
        #    newState = self.MC[self.state, :].searchsorted(r)

        #    if newState < self.nStates:
        #        self.state = newState

        #    trace[i] = self.state

        #return trace
        return StateMC(self.MC, N, startState)
            










