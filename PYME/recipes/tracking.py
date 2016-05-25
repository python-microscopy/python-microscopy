# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 19:59:15 2015

@author: david
"""
from .base import register_module, ModuleBase, Filter, Float, Enum, CStr, Bool, Int, View, Item#, Group
##from PYME.io.image import ImageStack
import numpy as np
import pandas as pd

from PYME.Analysis.Tracking import tracking
from PYME.Analysis.Tracking import trackUtils
from traits.api import on_trait_change

@register_module('TrackFeatures')
class TrackFeatures(ModuleBase):
    '''Take just certain columns of a variable'''
    inputMeasurements = CStr('measurements')
    outputTrackInfo = CStr('track_info')
    outputTracks = CStr('tracks') 
    
    
    features = CStr('x, y')    
    pNew = Float(0.2)
    r0 = Float(500)
    pLinkCutoff = Float(0.2)
    
    minTrackLength = Int(5)
    maxParticleSize = Float(20)
    
    def __init__(self, *args, **kwargs):
        ModuleBase.__init__(self, *args, **kwargs)
        
        self._tracker = None        
        #self.clumps = []
        
    @on_trait_change('pNew, r0, pLinkCutoff')    
    def OnParamChange(self):
        if not self._tracker == None:
            self._tracker.pNew=self.pNew
            self._tracker.r0 = self.r0
            self._tracker.linkageCuttoffProb = self.pLinkCutoff
            
    @on_trait_change('features')   
    def OnFeaturesChanged(self):
        self._tracker = None
        
    def Track(self, objects, newTracker=False):
        '''Track objects based on a given set of feature vectors'''        
        if (self._tracker == None) or not (len(self._tracker.t) == len(objects['t'])) or newTracker:
            featNames = [s.strip() for s in self.features.split(',')]
            
            def _calcWeights(s):
                fw = s.split('*')
                if len(fw) == 2:
                    return float(fw[0]), fw[1]
                else:
                    return 1.0, s
                    
            
            weightedFeats = [_calcWeights(s) for s in featNames]
            
            feats = np.vstack([w*np.array(objects[fn]) for w, fn in weightedFeats])
            
            self._tracker = tracking.Tracker(np.array(objects['t']), feats)
            
            self._tracker.pNew=self.pNew
            self._tracker.r0 = self.r0
            self._tracker.linkageCuttoffProb = self.pLinkCutoff

        for i in range(1, (objects['t'].max() + 1)):
            L = self._tracker.calcLinkages(i,i-1)
            self._tracker.updateTrack(i, L)
            
        clumpSizes = np.zeros_like(self._tracker.clumpIndex)
        
        for i in set(self._tracker.clumpIndex):
            ind = (self._tracker.clumpIndex == i)
            
            clumpSizes[ind] = ind.sum()
            
        trackVelocities = trackUtils.calcTrackVelocity(objects['x'], objects['y'], self._tracker.clumpIndex)
        
        clumpInfo = {'clumpIndex': self._tracker.clumpIndex, 'clumpSize': clumpSizes, 'trackVelocity': trackVelocities}
            

        pipe = {}
        pipe.update(clumpInfo)
        pipe.update(objects)
        pipe = pd.DataFrame(pipe)
        
        if 'mdh' in dir(objects):
            #propagate metadata
            pipe.mdh = objects.mdh 
            
        clumps = trackUtils.ClumpManager(pipe)
        
        clumps = [c for c in clumps.all if (c.nEvents > self.minTrackLength) and (c.featuremean['area'] < self.maxParticleSize)]

        clumpInfo = pd.DataFrame(clumpInfo)
        
        if 'mdh' in dir(objects):
            #propagate metadata
            clumpInfo.mdh = objects.mdh
        
        return clumpInfo, clumps
        
    def execute(self, namespace):       
        meas = namespace[self.inputMeasurements]
        clumpInfo, clumps = self.Track(meas, True)
        namespace[self.outputTrackInfo] = clumpInfo
        namespace[self.outputTracks] = clumps

    
    def TrackWithPipeline(self, pipeline):
        '''Support legacy interactive mode where we insert our results
        into the pipeline (used when called from PYME.DSView.modules.particleTracking)'''
        clumpInfo, clumps = self.Track(pipeline)
        
        pipeline.selectedDataSource.clumps = clumpInfo['clumpIndex']
        pipeline.selectedDataSource.setMapping('clumpIndex', 'clumps')
            
        pipeline.selectedDataSource.clumpSizes = clumpInfo['clumpSize']
        pipeline.selectedDataSource.setMapping('clumpSize', 'clumpSizes')
        
        pipeline.selectedDataSource.trackVelocities = clumpInfo['trackVelocity']
        pipeline.selectedDataSource.setMapping('trackVelocity', 'trackVelocities')
        
        #self.clumps = clumps        
        pipeline.clumps = clumps
        
        return clumps
