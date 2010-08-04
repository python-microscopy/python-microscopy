#!/usr/bin/python

##################
# remFitBuf.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import numpy
from PYME.Acquire.Hardware.Simulator.fakeCam import NoiseMaker
#import numpy as np

splitterFitModules = ['SplitterFitFR','SplitterFitQR','SplitterFitCOIR', 'BiplaneFitR', 'SplitterShiftEstFR', 'SplitterObjFindR', 'SplitterFitPsfIR']

from pylab import *
import copy

#[A, x0, y0, 250/2.35, dataMean.min(), .001, .001]

class fitTestJig:
    def __init__(self, metadata, fitModule):
        self.md = copy.copy(metadata)
        self.fitModule = fitModule
        self.md.tIndex = 0

        self.noiseM = NoiseMaker(EMGain=150)


    def runTests(self, params=[205, 0, 0, 250/2.35, 50, 0, 0], param_jit=[200, 90, 90, 30, 10, 0, 0], nTests=100):
        self.fitMod = __import__('PYME.Analysis.FitFactories.' + self.fitModule, fromlist=['PYME', 'Analysis','FitFactories']) #import our fitting module
        self.res = numpy.empty(nTests, self.fitMod.FitResultsDType)
        ps = zeros((nTests, len(params)))

        rs=5
        for i in range(nTests):
            p = array(params) + array(param_jit)*(2*rand(len(param_jit)) - 1)
            p[0] = abs(p[0])
            ps[i, :] = p
            self.data = self.fitMod.FitFactory.evalModel(p, self.md, roiHalfSize=rs)#, roiHalfSize= roiHalfWidth))

            self.d2 = self.noiseM.noisify(self.data)

            self.fitFac = self.fitMod.FitFactory(self.d2[:,:,None], self.md, background = 0)
            self.res[i] = self.fitFac.FromPoint(rs, rs, roiHalfSize=rs)

        
        return ps, self.res

    
