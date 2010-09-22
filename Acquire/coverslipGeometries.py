from scipy.optimize import fmin
import numpy as np




class geom:
    def __init__(self, fixedParameters):
        self.__dict__.update(fixedParameters)
        #self.fitableParameters = fitableParameters

    def Fit(self, x,y):
        self.fittedParameters = fmin(self.errFunc, self.genStartParams(x, y), args=(x, y))


class circGeom(geom):
    def errFunc(self, params, x, y):
        x0, y0, margin = params

        return (np.abs((x - x0)**2 + (y - y0)**2 - (self.r - margin)**2)).sum()

    def genStartParams(self, x,y):
        x0 = x.mean()
        y0 = y.mean()

        return [x0, y0, 0]


GEOMTYPES = {
'10mm round' : (circGeom, {'r':5})
}

def GetGeometry(type):
    cls, args = GEOMTYPES[type]
    return cls(args)



