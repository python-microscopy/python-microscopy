# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:05:41 2013

@author: david
"""
import os
from PYME.Analysis.LMVis.pipeline import Pipeline
from PYME.Analysis.processLogger import PL, TablesBackend
from PYME.Analysis import trackUtils

from PYME.Analysis.BleachProfile import kinModels
#turn graph display off
kinModels.USE_GUI = False


def analyseFile(filename):
    print filename
    PL.ExtendContext({'filename':filename})
    
    pipe = Pipeline(filename)
    
    #only look at first 7k frames
    pipe.filterKeys['t'] = (0, 7000)
    pipe.Rebuild()

    #find molecules appearing across multiple frames    
    trackUtils.findTracks(pipe, 'error_x', 2, 20)
    pipe.Rebuild()
    
    kinModels.fitDecay(pipe)
    kinModels.fitFluorBrightness(pipe)
    #kinModels.fitFluorBrightnessT(pipe)
    kinModels.fitOnTimes(pipe)
    
    pipe.CloseFiles()
    PL.PopContext()
    
if __name__ == '__main__':
    import sys
    
    #path to directory containing input files
    dataPath = sys.argv[1]
    
    #output file (hdf5 - use .h5p as extension)
    outfile = sys.argv[2]
    
    #set the logging backend to the PyTables file
    TB = TablesBackend(outfile)
    PL.SetBackend(TB)
    
    #walk the path and analyse all .h5r
    for (path, dirs, files) in os.walk(dataPath):
        for f in files:
            if os.path.splitext(f)[1] in ['.h5r']:
                filename = os.path.join(path, f)
                analyseFile(filename)