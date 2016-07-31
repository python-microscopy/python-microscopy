# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 19:59:15 2015

@author: david
"""
from .base import register_module, ModuleBase, Filter, Float, Enum, CStr, Bool, Int, View, Item#, Group
##from PYME.IO.image import ImageStack
import numpy as np
import pandas as pd

from PYME.Analysis.Tracking import tracking
from PYME.Analysis.Tracking import trackUtils
from traits.api import on_trait_change

from PYME.LMVis import inpFilt

@register_module('TrackFeatures')
class TrackFeatures(ModuleBase):
    """Take just certain columns of a variable"""
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
        """Track objects based on a given set of feature vectors"""
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
        """Support legacy interactive mode where we insert our results
        into the pipeline (used when called from PYME.DSView.modules.particleTracking)"""
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

@register_module('LoadSpeckles')
class LoadSpeckles(ModuleBase):
    """Loads Speckle data as used by the karatekin lab"""
    inputImage = CStr('', desc='The image to which the speckles file refers. Useful for determining speckle filename, series length, and voxelsize')
    speckleFilename = CStr('{DIRNAME}{SEP}{IMAGESTUB}speckles.csv',
                           desc='The filename of the speckle file. Anything in {} will be substituted with info from the input image.')
    #mappings = DictStrStr()
    outputName = CStr('speckles')
    leadFrames = Int(10, desc='The number of frames to add to the trace before the start of the speckle')
    followFrames = Int(50, desc='The number of frames to add to the trace after the end of the speckle')

    def execute(self, namespace):
        from PYME.IO.FileUtils import readSpeckle
        from PYME.IO import MetaDataHandler
        import os

        fileInfo = {'SEP' : os.sep}

        seriesLength = 100000

        mdh = MetaDataHandler.NestedClassMDHandler()

        if not self.inputImage == '':
            inp = namespace[self.inputImage]
            mdh.update(inp.mdh)
            seriesLength = inp.data.shape[2]

            try:
                fileInfo['DIRNAME'], fileInfo['IMAGENAME'] = os.path.split(inp.filename)
                fileInfo['IMAGESTUB'] = fileInfo['IMAGENAME'].split('MM')[0]
            except:
                pass

        speckleFN = self.speckleFilename.format(**fileInfo)

        specks = readSpeckle.readSpeckles(speckleFN)
        traces = readSpeckle.gen_traces_from_speckles(specks, leadFrames=self.leadFrames,
                                                      followFrames=self.followFrames, seriesLength=seriesLength)

        #turn this into an inputFilter object
        inp = inpFilt.recArrayInput(traces)

        #create a mapping to covert the co-ordinates in pixels to co-ordinates in nm
        map = inpFilt.mappingFilter(inp, x = 'x_pixels*%3.2f' % (1000*mdh['voxelsize.x']),
                                    y='y_pixels*%3.2f' % (1000 * mdh['voxelsize.y']))

        map.mdh = mdh

        namespace[self.outputName] = map


@register_module('ExtractTracks')
class ExtractTracks(ModuleBase):
    """Extract tracks from a measurement set with pre-assigned clump IDs"""
    inputMeasurements = CStr('speckles', desc='a set of particle positions already containing a clumpIndex column which identifies groups of particles')
    #outputTrackInfo = CStr('track_info', desc='a pandas data frame with clumpSize and trackVelocity information')
    outputTracks = CStr('tracks', desc='A clump / track manager object - aka the actual tracks')

    def execute(self, namespace):
        #print
        data = namespace[self.inputMeasurements]

        #print 'data.keys', data.keys()
        clumpIndex = data['clumpIndex']

        clumpSizes = np.zeros_like(clumpIndex)

        for i in set(clumpIndex):
            ind = (clumpIndex == i)

            clumpSizes[ind] = ind.sum()

        #trackVelocities = trackUtils.calcTrackVelocity(data['x'], data['y'], clumpIndex, data['t'])

        #clumpInfo = {'clumpSize': clumpSizes, 'trackVelocity': trackVelocities}

        clumps = trackUtils.ClumpManager(data)

        clumps = [c for c in clumps.all]

        #clumpInfo = pd.DataFrame(clumpInfo)

        if 'mdh' in dir(data):
            pass
            #propagate metadata
            #clumps.mdh = data.mdh
            #clumpInfo.mdh = data.mdh

        #namespace[self.outputTrackInfo] = clumpInfo
        namespace[self.outputTracks] = clumps
