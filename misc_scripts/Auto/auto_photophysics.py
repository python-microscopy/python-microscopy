# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:05:41 2013

@author: david
"""
import os
from PYME.LMVis.pipeline import Pipeline
from PYME.Analysis.processLogger import PL, TablesBackend, dictToRecarray
from PYME.Analysis.Tracking import trackUtils
import numpy as np

from PYME.Analysis.BleachProfile import kinModels
#turn graph display off
kinModels.USE_GUI = False


def analyseFile(filename):
    print(filename)
    seriesName = os.path.splitext(os.path.split(filename)[-1])[0]
    PL.ExtendContext({'seriesName':seriesName})
    try:
        pipe = Pipeline(filename)
    except RuntimeError:
        print(('Error opening %s' % filename))
        PL.PopContext()
        return
    
    #only look at first 7k frames
    pipe.filterKeys['t'] = (0, 7000)
    pipe.Rebuild()
    
    trackUtils.findTracks(pipe, 'error_x', 2, 20)
    pipe.Rebuild()

    extraParams = {}    
    extraParams['cycleTime'] = pipe.mdh['Camera.CycleTime']
    nPhot = kinModels.getPhotonNums(pipe.colourFilter, pipe.mdh)
    extraParams['MedianPhotons'] = np.median(nPhot)
    extraParams['MeanPhotons'] = np.mean(nPhot)
    extraParams['NEvents'] = len(nPhot)
    extraParams['MeanBackground'] = pipe['fitResults_background'].mean() - pipe.mdh['Camera.ADOffset']
    extraParams['MedianBackground'] = np.median(pipe['fitResults_background']) - pipe.mdh['Camera.ADOffset']
    extraParams['MeanClumpSize'] = pipe['clumpSize'].mean()
    extraParams['MeanClumpPhotons'] = (pipe['clumpSize']*nPhot).mean()
    
    PL.AddRecord('/Photophysics/ExtraParams', dictToRecarray(extraParams))
    
    kinModels.fitDecay(pipe)
    kinModels.fitFluorBrightness(pipe)
    #kinModels.fitFluorBrightnessT(pipe)

    #max_off_ts = [3,5,10,20,40]
    #max_off_ts = [20]

    #for ot in max_off_ts:
        #PL.ExtendContext({'otMax':ot})
        #find molecules appearing across multiple frames 
        
    kinModels.fitOnTimes(pipe)
        #PL.PopContext()
    
    pipe.CloseFiles()
     
    PL.PopContext()
	
def processDir(dataPath, outfile):
    #set the logging backend to the PyTables file
    TB = TablesBackend(outfile)
    PL.SetBackend(TB)
    
    #walk the path and analyse all .h5r
    for (path, dirs, files) in os.walk(dataPath):
        for f in files:
            if os.path.splitext(f)[1] in ['.h5r']:
                filename = os.path.join(path, f)
                analyseFile(filename)
				
if __name__ == '__main__':
    import sys
    
    #path to directory containing input files
    dataPath = sys.argv[1]
    
    #output file (hdf5 - use .h5p as extension)
    outfile = sys.argv[2]
    
    processDir(dataPath, outfile)
