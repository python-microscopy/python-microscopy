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

from PYME.Acquire.Hardware.Simulator.fakeCam import NoiseMaker

splitterFitModules = ['SplitterFitFR','SplitterFitQR','SplitterFitCOIR', 'BiplaneFitR', 'SplitterShiftEstFR', 'SplitterObjFindR', 'SplitterFitPsfIR']


import copy
from PYME.IO import MetaDataHandler
from PYME.Acquire.Hardware import EMCCDTheory
from scipy import optimize
import numpy as np
from PYME.localization import remFitBuf

def emg(v, rg):
    return (EMCCDTheory.M((80. + v)/(255 + 80.), 6.6, -70, 536, 2.2) - rg)**2
    
def IQR(x):
    return np.percentile(x, 75) - np.percentile(x, 25)

#[A, x0, y0, 250/2.35, dataMean.min(), .001, .001]

class fitTestJig(object):
    """
    A class for testing fits.

    Reads both simulation and analysis settings out of a metadata file, looking for the special metadata keys:
      "Test.DefaultParams" - a vector containing the mean values of each  model parameter to use when generating the
                             test data.
      "Test.ParamJitter" - a vector containing the magnitude of variation to add to each parameter. Each test image is
                           generated with a different the model parameters randomly jittered in the range
                           [DefaultParams - ParamJitter : DefaultParams + ParamJitter]. To generate data with a fixed
                           set of parameter values, set ParamJitter to zeros
      "Test.ROISize"    - the half-size fo the ROI to simulate
      "Test.SimModule" - the name of the module used to simulate the data (if different from the analysis module)
      "Test.Background" - an additional background to add before noise generation and then subtract, to be used with fits
                          which do not have a background parameter and rely on the running background subtraction to
                          provide an essentially background free image (some of the 3D fits).
      "Test.PSFFile" - the name of a PSF file to use for simulation if not simulating with the same PSF which is used
                       for analysis [3D fits].
      "Test.VarianceMapID" - variance map to use for simulation (if different from analysis)
      "Test.DarkMapID" - dark map to use for simulation (if different from analysis)
      "Test.FlatfieldMapID" - dark map to use for simulation (if different from analysis)

    """
    def __init__(self, metadata, fitModule = None):
        """
        Initialize the fit test jig from metadata file

        Parameters
        ----------
        metadata - PYME.IO.MetadataHandler.MetaDataHandler object
            The metadata containing simulation and analysis settings. See custom 'Test.XX' keys in class description
        fitModule - sting [optional, deprecated]
            The name of the fit module to use. This overrides the settings in the metadata, but is only retained for
            backwards compatibility. You should set the fit module in the metadata.
        """
        self.md = copy.copy(metadata)
        if fitModule == None:
            self.fitModule = self.md.getEntry('Analysis.FitModule')
        else:
            self.fitModule = fitModule
        self.md.tIndex = 0
        
        if 'Test.SimModule' in self.md.getEntryNames():
            self.simModule = self.md['Test.SimModule']
        else:
            self.simModule = self.fitModule
        
        self.bg = 0
        if 'Test.Background' in self.md.getEntryNames():
            self.bg = float(self.md['Test.Background'])

        self.rs=self.md.getOrDefault('Test.ROISize', 7)

        self._prepSimulationCameraMaps()

        #by still including emGain estimation etc ... we can use the same code for sCMOS and EMCCD estimation. This will
        #evaluate to 1 in the sCMOS case
        emGain = optimize.fmin(emg, 150, args=(float(self.md.Camera.TrueEMGain),))[0]

        self.noiseM = NoiseMaker(floor=self.dark, readoutNoise=np.sqrt(self.variance),
                                 ADGain= self.md['Camera.ElectronsPerCount']*self.gain,
                                 background=self.bg, QE=1.0, EMGain=emGain)

    def _prepSimulationCameraMaps(self):
        md2 = MetaDataHandler.NestedClassMDHandler(self.md)
        self.cameraMaps = remFitBuf.CameraInfoManager()

        #if we have set different camera maps for simulation, use those, otherwise use the ones defined for analysis
        try:
            md2['Camera.VarianceMapID'] = md2['Test.VarianceMapID']
        except (KeyError, AttributeError):
            pass

        try:
            md2['Camera.DarkMapID'] = md2['Test.DarkMapID']
        except (KeyError, AttributeError):
            pass

        try:
            md2['Camera.FlatfieldMapID'] = md2['Test.FlatfieldMapID']
        except (KeyError, AttributeError):
            pass

        #get camera maps using metadata
        self.dark = self.cameraMaps.getDarkMap(md2)
        self.variance = self.cameraMaps.getVarianceMap(md2)
        self.gain = 1.0 / self.cameraMaps.getFlatfieldMap(md2)

        #crop the maps to the ROI size
        _roiWidth = 2 * self.rs + 1
        try:
            self.dark = self.dark[:_roiWidth, :_roiWidth]
        except TypeError:
            #if no gain maps are specified (i.e. EMCCD case), self.dark etc ... will be scalar
            pass

        try:
            self.variance = self.variance[:_roiWidth, :_roiWidth]
        except TypeError:
            #if no gain maps are specified (i.e. EMCCD case), self.dark etc ... will be scalar
            pass

        try:
            self.gain = self.gain[:_roiWidth, :_roiWidth]
        except TypeError:
            #if no gain maps are specified (i.e. EMCCD case), self.dark etc ... will be scalar
            pass



    @classmethod
    def fromMDFile(cls, mdfile):
        """
        Create a new fit test jig from a metadata file

        Parameters
        ----------
        mdfile - string
            the filename of the metadata file to use. The file should be in PYMEs 'SimpleMDHandler' (*.md) format.

        Returns
        -------
        fitTestJig instance

        """
        return cls(MetaDataHandler.SimpleMDHandler(mdfile))


    def runTests(self, params=None, param_jit=None, nTests=100):
        """
        Simulate and fit multiple single molecules. The results are stored in the class itself and can be accessed by
        other methods.

        Parameters
        ----------
        params - array
            The mean value of the parameters. Overrides settings in metadata if provided
        param_jit - array
            The amount of parameter jitter. Overrides settings in metadata if provided
        nTests - int
            The number of molecules to simulate

        Returns
        -------

        self.sim_params contains the parameters used for simulation
        self.results contains the full fit results
        self.result_params contains the fitted params
        self.d2 is a list of the simulated data frames

        NB: the x and y params in both results and result_params will be offset by half the region of interest size. i.e.
        simulation co-ordinates are taken from the ROI centre and fitting coordinates are taken from the top left corner.

        """
        if not params:
            params = self.md['Test.DefaultParams']
        if not param_jit:
            param_jit = self.md['Test.ParamJitter']

        #load the modules used for simulation and fitting
        self.fitMod = __import__('PYME.localization.FitFactories.' + self.fitModule, fromlist=['PYME', 'localization', 'FitFactories']) #import our fitting module
        self.simMod = __import__('PYME.localization.FitFactories.' + self.simModule, fromlist=['PYME', 'localization', 'FitFactories']) #import our simulation

        #generate empty arrays for parameters and results
        self.res = np.empty(nTests, self.fitMod.FitResultsDType)
        ps = np.zeros((nTests, len(params)), 'f4')

        #create a copy of our metadata to use for simulating the data. This can use a different PSF to the analysis
        #md2 = copy.copy(self.md)
        md2 = MetaDataHandler.NestedClassMDHandler(self.md)
        if 'Test.PSFFile' in self.md.getEntryNames():
            md2['PSFFile'] = self.md['Test.PSFFile']
        
        self.d2 = []

        #generate our data
        ####################
        for i in range(nTests):
            p = np.array(params) + np.array(param_jit)*(2*np.random.rand(len(param_jit)) - 1)
            p[0] = abs(p[0])
            ps[i, :] = p
            self.data, self.x0, self.y0, self.z0 = self.simMod.FitFactory.evalModel(p, md2, roiHalfSize=self.rs)#, roiHalfSize= roiHalfWidth))
            self.d2.append(self.md.Camera.ADOffset + 1*(self.noiseM.noisify(self.data) - self.md.Camera.ADOffset))

            
        #calculate our background
        #bg = self.bg*1.0/(self.md.Camera.TrueEMGain/self.md.Camera.ElectronsPerCount) + self.md.Camera.ADOffset
        #print((bg, self.noiseM.getbg()))
        #bg = self.noiseM.getbg()
        bg = remFitBuf.cameraMaps.getDarkMap(self.md)

        #calculate the fits
        ###################
        for i in range(nTests):
            self.sigma = remFitBuf.fitTask.calcSigma(self.md, np.atleast_3d(self.d2[i] - self.md['Camera.ADOffset']))
            self.fitFac = self.fitMod.FitFactory(np.atleast_3d(self.d2[i]), self.md, background = bg, noiseSigma = self.sigma)
            self.res[i] = self.fitFac.FromPoint(self.rs, self.rs, roiHalfSize=self.rs)

        
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

    #these properties are just making the results available under more sensible names
    @property
    def sim_params(self):
        return self.ps

    @property
    def results(self):
        return self.res

    @property
    def result_params(self):
        return self.res['fitResults']
            
            
    def error(self, varName):
        """
        Calculate the error (fitted - expected) for the parameter given by varName

        Parameters
        ----------
        varName - string
            the name of the parameter to calculate the error for

        Returns
        -------
        array of differences between fitted and model values

        """
        xv = self.ps[varName].ravel()
        yv = self.res['fitResults'][varName]
        if hasattr(self, varName):
            yv = yv + self.__getattribute__(varName)
            
        return yv - xv


    def plotRes(self, varName):
        """
        Plot a scatter plot of the fitted vs the simulated values.

        Note that this version also attempts to plot the starting parameters used in the fit, so will not work with
        fits which do not record their starting parameters (e.g. latGaussFitFR)

        Parameters
        ----------
        varName - string
            the name of the parameter to plot

        Returns
        -------

        """
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
        """
        Plot a scatter plot of the fitted vs the simulated values (simple version).

        This just plots the fitted vs model values without errorbars or starting positions, so should work for all fits.

        Parameters
        ----------
        varName - string
            the name of the parameter to plot

        Returns
        -------

        """
        #print self.ps
        #from pylab import *
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









    
