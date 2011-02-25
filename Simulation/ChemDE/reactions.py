import numpy as np
from scipy.integrate import ode
import sympy

class Reaction:
    def __init__(self, equation, k_forward=0, k_reverse=0, name=None):
        self.reaction_equation = equation
        self.k_forward = k_forward
        self.k_reverse = k_reverse
        self.name = name

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
            name = '(%3.5f*%s)' % ties[name]

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

        #return DEs
        ##precompile the equations for speed
        cDEs = {}
        for key, value in DEs.items():
            cDEs[key] = str(sympy.simplify(sympy.sympify(value)))#, '/tmp/test1', 'eval')

        return cDEs

    def GenerateGradAndJacCode(self):
        DEs = self.getDEs()

        gradcode = '\n'.join(['%s = v["%s"]' % (n,n) for n in self.dtype.names])
        jaccode = '' + gradcode

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


    def solve(self, tvals):
        
        o = ode(self.GradFcn, self.JacFcn).set_integrator('vode', method='bdf', with_jacobian=True)
        o.set_initial_value(self.initialConditions.view('f8'), 0)
        #o.set_f_params(self.dtype, self.getDEs(), self.constants, self.stimulae)

        out = np.zeros(len(tvals), self.dtype)

        for i, t in enumerate(tvals):
            o.integrate(t)
            if o.successful():
                out[i] = o.y
            else:
                raise RuntimeError('Integration Failed')

        return out




