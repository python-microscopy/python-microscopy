# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 00:55:38 2015

@author: david
"""

import numpy as np
import pylab as pl
from PYME.DSView import image
import PYME.Analysis.trackUtils as trackUtils

def plotEvent(clump, pipeline, rawData = None):
    pl.figure()
    
    vs = 1e3*pipeline.mdh['voxelsize.x']
    xp = int(clump['x'].mean()/vs)
    yp = int(clump['y'].mean()/vs)
    
    pl.subplot(311)
    pl.plot(clump['t'], clump['Ag']/clump['Ag'].max())
    #ym = (2*np.pi*clump['fitResults_Ag']*(clump['fitResults_sigma']/vs)**2)
    #pl.grid()
    #pl.ylim(0, ym)
    #pl.twinx()
    pl.plot(clump['t'], clump['Ar']/clump['Ar'].max(), alpha=.5)
    #pl.plot(clump['t'], 2*np.pi*clump['fitResults_Ar']*(clump['fitResults_sigmag']/vs)**2)
    

    #ym = (2*np.pi*clump['fitResults_Ag']*(clump['fitResults_sigma']/vs)**2).max()
    #ym2 = (2*np.pi*clump['fitResults_Ar']*(clump['fitResults_sigmag']/vs)**2).max()
    
    #pl.ylim(0, max(ym, ym2))
    pl.ylim(0, 1)
    pl.grid()
    
    pl.ylabel('Intensity [a.u.]')
    pl.xlabel('Time [frames]')
    pl.legend(['Lipid', 'Cargo'])
    
    pl.title('clumpIndex=%d, position = (%d, %d) px' % (clump['clumpIndex'][0], xp, yp))
    
    
    pl.subplot(312)
    pl.plot(clump['t'], clump['fitResults_sigma'])
    pl.plot(clump['t'], clump['fitResults_Ar'], alpha=.5)
    
    pl.ylim(0, 1e3)
    pl.grid()
    
    pl.ylabel('Fit $\sigma$ [nm]')
    pl.xlabel('Time [frames]')
    pl.legend(['Lipid', 'Cargo'])
    
    pl.subplot(615)
    #try:
    rawImg = image.openImages.items()[-1][1]
    
    fnums = np.linspace(clump['t'].min(), clump['t'].max(), 10).astype('i')
    frames = np.hstack([rawImg.data[(xp-10):(xp+10), (yp-10):(yp + 10), fi].squeeze() for fi in fnums])
    pl.imshow(frames, interpolation='nearest', cmap='hot')
    
    pl.subplot(616)
    xp = int((clump['x'].mean() - pipeline.mdh['chroma.dx'](0,0))/vs)
    yp = int((clump['y'].mean() - pipeline.mdh['chroma.dy'](0,0))/vs) + 256
    frames = np.hstack([rawImg.data[(xp-10):(xp+10), (yp-10):(yp + 10), fi].squeeze() for fi in fnums])
    pl.imshow(frames, interpolation='nearest', cmap='hot')
    #except:
    #    pass
    
    
def prepPipeline(pipeline):
    #we fit peak intensity - define mappings which are propotional to integrated intensity instead
    pipeline.mapping.a_norm = 2*np.pi/(1e3*pipeline.mdh['voxelsize.x'])**2
    pipeline.mapping.setMapping('Ag', 'a_norm*fitResults_Ag*fitResults_sigma**2')
    pipeline.mapping.setMapping('Ar', 'a_norm*fitResults_Ar*fitResults_sigmag**2')
    
    #choose 'plausible' fits    
    pipeline.filterKeys = {'A': (5, 20000), 'sig': (95.0, 1500)}
    pipeline.Rebuild()
    
    #perform tracking
    #this is simple event chaining which works well for well separated events
    #here we connect everything which is within 500 nm, allowing temporal gaps of up to 5 frames 
    trackUtils.findTracks(pipeline, '1.0',500, 5)
    
    #tracking introduces two new variables into the pipeline - clumpIndex (which
    #is a unique identifier for the particular track), and clumpSize (which is the length of the track)    
    
    #select only the tracks which have more than 100 frames
    pipeline.filterKeys['clumpSize'] =  (100.0, 100000.0)
    pipeline.Rebuild()
    

def selectAndPlotEvents(pipeline, outputdir='/Users/david/FusionAnalysis'):
    import os
    import pandas as pd    
    #now iterate through our clumps
    clumpIndices = list(set(pipeline['clumpIndex']))
    
    clumps = [pipeline.clumps[i] for i in clumpIndices]
    
    #do another level of filtering - fusion events expand, so we're looking for larger than normal
    #sigma in the lipid channel. We can also add a constraint on the mean intensity as proper docking
    #and fusion events are brighter than a lot of the point-like rubbish
    #We do this here, so we can filter on the aggregate behaviour of a track and be more resiliant against
    #noise.
    
    filteredClumps = [c for c in clumps if (c['Ag'].mean() > 2000) and (c['fitResults_sigma'].mean() > 300)]
    
    outputDir = os.path.join(outputdir, os.path.split(pipeline.filename)[1]) 
    os.makedirs(outputDir)
    
    for c in filteredClumps:
        plotEvent(c, pipeline)
        pl.savefig('%s/track%d.pdf' % (outputDir, c['clumpIndex'][0]))
        
        d = {}
        d.update(c)
        pd.DataFrame(d).to_csv('%s/track%d.csv' % (outputDir, c['clumpIndex'][0]))
    
    
    
    
    