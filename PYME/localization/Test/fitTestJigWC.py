#!/usr/bin/python

##################
# remFitBuf.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

"""
Test a fit - in general use FitTestJigWC instead

"""


import numpy
from PYME.Acquire.Hardware.Simulator.fakeCam import NoiseMaker
#import numpy as np

splitterFitModules = ['SplitterFitFR','SplitterFitQR','SplitterFitCOIR', 'BiplaneFitR', 'SplitterShiftEstFR', 'SplitterObjFindR', 'SplitterFitPsfIR']

#from pylab import *
import copy
from PYME.IO import MetaDataHandler
from PYME.Acquire.Hardware import EMCCDTheory
from scipy import optimize
import numpy as np

def emg(v, rg):
    return (EMCCDTheory.M((80. + v)/(255 + 80.), 6.6, -70, 536, 2.2) - rg)**2
    
def IQR(x):
    return np.percentile(x, 75) - np.percentile(x, 25)

#[A, x0, y0, 250/2.35, dataMean.min(), .001, .001]

class fitTestJig(object):
    def __init__(self, metadata, fitModule = None):
        self.md = copy.copy(metadata)
        if fitModule is None:
            self.fitModule = self.md.getEntry('Analysis.FitModule')
        else:
            self.fitModule = fitModule
        
        self.md['tIndex'] = 0
        
        self.simModule = self.md.getOrDefault('Test.SimModule', self.fitModule)     
        self.bg = float(self.md.getOrDefault('Test.Background', 0.0))

        emGain = optimize.fmin(emg, 150, args=(float(self.md['Camera.TrueEMGain']),))[0]

        self.noiseM = NoiseMaker(EMGain=emGain, floor=self.md['Camera.ADOffset'], background=self.bg, QE=1.0)


    @classmethod
    def fromMDFile(cls, mdfile):
        return cls(MetaDataHandler.SimpleMDHandler(mdfile))


    def runTests(self, params=None, param_jit=None, nTests=100):
        from PYME.localization.FitFactories import import_fit_factory
        if not params:
            params = self.md['Test.DefaultParams']
        if not param_jit:
            param_jit = self.md['Test.ParamJitter']
            
        self.fitMod = import_fit_factory(self.fitModule) #import our fitting module
        self.simMod = import_fit_factory(self.simModule) #import our simulation
        self.res = np.empty(nTests, self.fitMod.FitResultsDType)
        ps = np.zeros((nTests, len(params)), 'f4')

        rs=self.md.getOrDefault('Test.ROISize', 7)
        
        
        
        md2 = copy.copy(self.md)
        if 'Test.PSFFile' in self.md.getEntryNames():
            md2['PSFFile'] = self.md['Test.PSFFile']
        
        self.d2 = []
        
        for i in range(nTests):
            p = np.array(params) + np.array(param_jit)*(2*np.random.rand(len(param_jit)) - 1)
            print('p: %s' % repr(p))
            p[0] = abs(p[0])
            ps[i, :] = p
            self.data, self.x0, self.y0, self.z0 = self.simMod.FitFactory.evalModel(p, md2, roiHalfSize=rs)#, roiHalfSize= roiHalfWidth))
            
            #print self.data.shape
            
            #from PYME.DSView import View3D
            #View3D(self.data)

            self.d2.append(self.md['Camera.ADOffset'] + 1*(self.noiseM.noisify(self.data) - self.md['Camera.ADOffset']))
            #print self.d2.min(), self.d2.max(), self.data.min(), self.data.max()
            
            #print self.d2.shape
            
        bg = self.bg*1.0/(self.md['Camera.TrueEMGain']/self.md['Camera.ElectronsPerCount']) + self.md['Camera.ADOffset']
        
        print((bg, self.noiseM.getbg()))
        
        bg = self.noiseM.getbg()
            
        #print bg, self.md.Camera.ADOffset
        for i in range(nTests):
            self.fitFac = self.fitMod.FitFactory(np.atleast_3d(self.d2[i]), self.md, background = bg)
            self.res[i] = self.fitFac.FromPoint(rs, rs, roiHalfSize=rs)

        
        self.ps = ps.view(np.dtype(self.simMod.FitResultsDType)['fitResults'])
        
        #self.calcMEs()
        #return ps.view(self.res['fitResults'].dtype), self.res
        
    def calcMEs(self):
        for varName in self.ps.dtype.names:
            yv = self.res['fitResults'][varName]
            if hasattr(self, varName):
                yv = yv + self.__getattribute__(varName)
                
            me = ((self.ps[varName].ravel() - yv)**2).mean()
            print(('%s: %3.2f' % (varName, me)))
            
            
    def error(self, varName):
        xv = self.ps[varName].ravel()
        yv = self.res['fitResults'][varName]
        if hasattr(self, varName):
            yv = yv + self.__getattribute__(varName)
            
        return yv - xv


    def plotRes(self, varName):
        #print self.ps
        #from pylab import *
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14,5))
        #print varName
        xv = self.ps[varName].ravel()
        
        sp = self.res['startParams'][varName]
        yv = self.res['fitResults'][varName]

        if hasattr(self, varName):
            sp = sp + self.__getattribute__(varName)
            yv = yv + self.__getattribute__(varName)

        err = self.res['fitError'][varName]

        plt.subplot(121)

        
       # plot(xv, sp, '+', label='Start Est')
        plt.errorbar(xv, yv, err, fmt='r.', label='Fitted')
        plt.plot(xv, yv, 'xg', label='Fitted')
        plt.plot([xv.min(), xv.max()], [xv.min(), xv.max()])

        plt.ylim((yv - np.maximum(err, 0)).min(), (yv + np.maximum(err, 0)).max())
        plt.legend()

        plt.title(varName)
        plt.xlabel('True Position')
        plt.ylabel('Estimated Position')

        plt.subplot(122)
        dv = yv - xv
        iq = IQR(dv)
        print('Mean: %f, std. dev: %f, IQR: %f' % (dv.mean(), dv.std(), iq))
        plt.hist(dv, np.linspace(dv.mean()-3*iq, dv.mean() + 3*iq))
        plt.xlabel('Position Error [nm]')
        plt.ylabel('Frequency')
        
        
    def plotResSimp(self, varName):
        #print self.ps
        import matplotlib.pyplot as plt
        plt.figure()
        #print varName
        xv = self.ps[varName].ravel()
        
        #sp = self.res['startParams'][varName]
        yv = self.res['fitResults'][varName]

        if hasattr(self, varName):
            #sp = sp + self.__getattribute__(varName)
            yv = yv + self.__getattribute__(varName)

        #err = self.res['fitError'][varName]

        plt.plot([xv.min(), xv.max()], [xv.min(), xv.max()])
        #plot(xv, sp, '+', label='Start Est')
        plt.plot(xv, yv, 'x', label='Fitted')

        plt.ylim((yv).min(), (yv).max())
        #legend()

        plt.title(varName)
        plt.xlabel('True Position')
        plt.ylabel('Estimated Position')









    
