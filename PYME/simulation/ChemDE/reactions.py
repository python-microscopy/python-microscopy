import numpy as np
from scipy.integrate import ode
import sympy

class Reaction:
    def __init__(self, equation, k_forward=0, k_reverse=0, name=None, catalysts=[]):
        self.reaction_equation = equation
        self.k_forward = k_forward
        self.k_reverse = k_reverse
        self.name = name
        self.catalysts = catalysts

        self._parse(equation)

    def _parse(self, equation):
        reagents, products = equation.split('<->')

        reagents = reagents.split(' + ')
        products = products.split(' + ')

        self.reagents = {}
        for r in reagents:
            reag, mol = self._parseReagent(r)
            self.reagents[reag] = mol

        self.products = {}
        for r in products:
            reag, mol = self._parseReagent(r)
            self.products[reag] = mol

    def getSpecies(self):
        return list(set(self.reagents.keys() + self.products.keys()))

    def _molPow(self, spec, ties = {}):
        name, mol = spec

        if name in ties.keys(): #if we've got a tie, replace with the tied variable
            name = '(%s*%s)' % ties[name]

        #print spec

        if mol == 1:
            return name
        else:
            return name + '**%d' % mol

    def _parseReagent(self,str):
        '''cleanup reagent name and extract molarity'''
        str = str.strip() #remove leading & tailing whitespace

        reag, mol = self._stripNum(str)

        #dump any characters we don't want
        reag = ''.join([c for c in reag if c.isdigit() or c.isalpha() or c in '_'])

        return reag, mol


    def _stripNum(self,str):
        '''extract a leading multiplier'''
        numStr = ''

        while str[0].isdigit():
            numStr += str[0]
            str = str[1:]

        if numStr == '':
            num = 1
        else:
            num = int(numStr)

        return str, num

    def getDEs(self, ties = {}):
        DEs = {}

        if self.name == None:
            forward_expr = ('%3.5f*' % self.k_forward) +  '*'.join([self._molPow(spec, ties) for spec in self.reagents.items()])
            reverse_expr = ('%3.5f*' % self.k_reverse) +  '*'.join([self._molPow(spec, ties) for spec in self.products.items()])
        else:
            forward_expr = ('k_%sf*' % self.name) +  '*'.join([self._molPow(spec, ties) for spec in self.reagents.items()])
            reverse_expr = ('k_%sr*' % self.name) +  '*'.join([self._molPow(spec, ties) for spec in self.products.items()])

        for reag, mol in self.reagents.items():
            s = '-%s + %s' % (forward_expr, reverse_expr)
            if mol > 1:
                s = '%d*(%s)' % (mol, s)

            if reag in ties.keys():
                sc, reag = ties[reag]
                #s = '(%s)/%3.6f' % (s, sc)

            if reag in DEs.keys(): #already have a term on LHS
                s = DEs[reag] + ' + ' + s

            DEs[reag] = s

        for reag, mol in self.products.items():
            s = '%s - %s' % (forward_expr, reverse_expr)
            if mol > 1:
               s = '%d*(%s)' % (mol, s)
               

            if reag in ties.keys():
                sc, reag = ties[reag]
                #s = '(%s)/%3.6f' % (s, sc)

            if reag in DEs.keys(): #already have a term on LHS
                s = DEs[reag] + ' + ' + s

            DEs[reag] = s

        return DEs
        
    def getRateInfo(self,species, ties, constants):
        n_forward = np.zeros(len(species))
        n_backward = np.zeros_like(n_forward)
        m_forward = np.zeros_like(n_forward)
        m_backward = np.zeros_like(n_forward)
        
        kf = self.k_forward
        kb = self.k_reverse
        
        #print ties, constants
        
        for sp, mol in self.reagents.items():
            if sp in constants.keys():
                kf *= constants[sp]
            else:
                if sp in ties.keys():
                    kf*=constants[ties[sp][0]]
                    sp = ties[sp][1]
                m_forward[species.index(sp)] = mol
                n_forward[species.index(sp)] = -mol
                n_backward[species.index(sp)] = mol
            
        for sp, mol in self.products.items():
            if sp in constants.keys():
                kb *= constants[sp]
            else:
                if sp in ties.keys():
                    kb*=constants[ties[sp][0]]
                    sp = ties[sp][1]
                m_backward[species.index(sp)] += mol
                n_forward[species.index(sp)] += mol
                n_backward[species.index(sp)] += -mol
            
        return n_forward, n_backward, m_forward, m_backward, kf, kb
                
            
        
        
        


def gradFunction(t, y, dtype, DEs, constants, stimulae):
    v = y.view(dtype)
    for n in dtype.names:
        locals()[n] = v[n]
        
    locals().update(constants)

    for key, fcn in stimulae.items():
        locals()[key] = fcn(t)

    #print locals()

    res = 0*y
    r = res.view(dtype)

    for n in dtype.names:
        r[n] = eval(DEs[n])

    return res

class Stimulus:
    '''Piecewise constant stimulus function'''
    def __init__(self, initValue, times, values):
        self.initValue = initValue
        self.times = np.array(times)
        self.values = np.array(values)

    def __call__(self, t):
        i = self.times.searchsorted(t, side='right')

        if np.isscalar(t):
            if i == 0:
                return self.initValue
            else:
                return self.values[i-1]
        else:
            #print i.dtype
            vals = self.initValue + 0*np.array(t)

            vals[i> 0] = self.values[i[i>0] - 1]

            return vals

        


class System:
    def __init__(self, reactions, constants={}, stimulae={}, ties={}):
        self.reactions = reactions
        self.constants = constants
        self.stimulae = stimulae

        #species which are in a tight equillibrium - treat as one in reactions
        #to have the format ties['VarToReplace'] = (scale, 'OtherVar')
        self.ties = ties 

        spec = []
        for r in reactions:
            spec += r.getSpecies()
            
        self.species = list(set(spec))
        #remove any species we're holding constant
        self.species = [s for s in self.species if not (s in constants.keys() or s in stimulae.keys() or s in ties.keys())]

        #default to zero intial conditions
        self.dtype = np.dtype({'names':self.species, 'formats':['f8' for i in range(len(self.species))]})
        self.initialConditions = np.zeros(1, dtype=self.dtype)

        #we need to stop (and restart) the integration at any discontinuities
        #search stimlulae for jumps and to list
        self.discontinuities = []
        
        for stim in self.stimulae.values():
            self.discontinuities += list(stim.times)

        self.discontinuities = np.array(self.discontinuities)
        self.discontinuities.sort()

        #self.GenerateGradAndJacCode()

    def getDEs(self):
        DEs = {}
        #initialise all our DEs to an empty list
        for spec in self.species:
            DEs[spec] = []
            
        for r in self.reactions:
            reactDEs = r.getDEs(self.ties)
            
            for key, value in reactDEs.items():
                if key in self.species:
                    DEs[key].append(value)

        for spec in self.species:
            DEs[spec] = ' + '.join(DEs[spec])

        return DEs
        ##simplify the equations for speed
        #cDEs = {}
        #for key, value in DEs.items():
        #    cDEs[key] = str(sympy.simplify(sympy.sympify(value)))#, '/tmp/test1', 'eval')

        #return cDEs

    def GenerateGradAndJacCode(self):
        DEs = self.getDEs()

        #expand values
        gradcode = '\n'.join(['%s = v["%s"]' % (n,n) for n in self.dtype.names])
        jaccode = '' + gradcode #copy value expansion code to use for jacobian

        #simplify DEs
        for key, value in DEs.items():
            i = self.dtype.names.index(key)
            de = sympy.simplify(sympy.sympify(value))
            gradcode += '\nres[%d] = %s' %(i, de)
            for j, n in enumerate(self.dtype.names):
                dde = de.diff(n)
                if not dde == 0:
                    jaccode += '\njac[%d,%d] = %s' % (i, j, dde)

        #print gradcode
        #print jaccode

        self.gradCode = compile(gradcode, '/tmp/test1', 'exec')
        self.jacCode = compile(jaccode, '/tmp/test1', 'exec')


    def GradFcn(self, t, y):
        v = y.view(self.dtype)
        locals().update(self.constants)
        for key, fcn in self.stimulae.items():
            locals()[key] = fcn(t)

        res = 0*y
        #r = res.view(self.dtype)

        exec(self.gradCode)

        return res

    def JacFcn(self, t, y):
        v = y.view(self.dtype)
        locals().update(self.constants)
        for key, fcn in self.stimulae.items():
            locals()[key] = fcn(t)

        #print
        jac = np.zeros((y.size, y.size))

        exec(self.jacCode)

        return jac


    def solve(self, tvals, grad=False):
        
        o = ode(self.GradFcn, self.JacFcn).set_integrator('vode', method='bdf', with_jacobian=True)
        o.set_initial_value(self.initialConditions.view('f8'), 0)
        #o.set_f_params(self.dtype, self.getDEs(), self.constants, self.stimulae)

        out = np.zeros(len(tvals), self.dtype)
        if grad:
            gr = np.zeros(len(tvals), self.dtype)

        disc_i = 0

        for i, t in enumerate(tvals):
            ndisc_i =  self.discontinuities.searchsorted(t, side='right')
            if not ndisc_i == disc_i:
                #print 'Restarting integrator'
                tDisc = self.discontinuities[ndisc_i-1]
                o.integrate(tDisc)
                o.set_initial_value(o.y, tDisc)
                disc_i = ndisc_i
                
            o.integrate(t)
            if o.successful():
                out[i] = o.y
                if grad:
                    gr[i] = o.jac(o.t, o.y)
            else:
                raise RuntimeError('Integration Failed')

        if grad:
            return out, gr
        else:
            return out




