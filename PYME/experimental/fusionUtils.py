# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 00:55:38 2015

@author: david
"""

import numpy as np
# import pylab as pl
import matplotlib.pyplot as pl
from PYME.IO import image
import PYME.Analysis.Tracking.trackUtils as trackUtils
import json

def plotEvent(clump, pipeline, rawData = None, plotRaw = True):
    pl.figure()
    
    vs = pipeline.mdh.voxelsize_nm.x
    xp = int(clump['x'].mean()/vs)
    yp = int(clump['y'].mean()/vs)

    ind1 = (clump['fitError_Ag'] < 1e3)*(clump['fitError_Ag'] > 0)
    ind2 = (clump['fitError_Ar'] < 1e3)*(clump['fitError_Ar'] > 0)*(clump['Ar'] > 100)    
    
    pl.subplot(311)
    #pl.plot(clump['t'], clump['Ag']/clump['Ag'][ind1].max(), 'g')
    pl.plot(clump['t'][ind1], clump['Ag'][ind1]/clump['Ag'][ind1].max(), 'b')
    #ym = (2*np.pi*clump['fitResults_Ag']*(clump['fitResults_sigma']/vs)**2)
    #pl.grid()
    #pl.ylim(0, ym)
    #pl.twinx()
    #pl.plot(clump['t'], clump['Ar']/clump['Ar'][ind2].max(), 'b', alpha=.5)
    pl.plot(clump['t'][ind2], clump['Ar'][ind2]/clump['Ar'][ind2].max(), 'g', alpha=.5)
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
    pl.plot(clump['t'][ind1], clump['fitResults_sigma'][ind1])
    pl.plot(clump['t'][ind2], clump['fitResults_sigmag'][ind2], alpha=.5)
    
    pl.ylim(0, 1e3)
    pl.grid()
    
    pl.ylabel('Fit $\sigma$ [nm]')
    pl.xlabel('Time [frames]')
    pl.legend(['Lipid', 'Cargo'])
    
    pl.subplot(615)
    if plotRaw:
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


def plotEvent2(clump, pipeline, rawData = None, plotRaw = True):
    pl.figure()
    
    vs = pipeline.mdh.voxelsize_nm.x
    xp = int(clump['x'].mean()/vs)
    yp = int(clump['y'].mean()/vs)

    ind1 = (clump['fitError_Ag'] < 1e3)*(clump['fitError_Ag'] > 0)
    ind2 = (clump['fitError_Ar'] < 1e3)*(clump['fitError_Ar'] > 0)*(clump['Ar'] > 100)    
    
    pl.subplot(311)
    #pl.plot(clump['t'], clump['Ag']/clump['Ag'][ind1].max(), 'g')
    pl.plot(clump['t'][ind1], clump['Ag'][ind1]/clump['Ag'][ind1].max(), 'b')
    #ym = (2*np.pi*clump['fitResults_Ag']*(clump['fitResults_sigma']/vs)**2)
    #pl.grid()
    #pl.ylim(0, ym)
    #pl.twinx()
    #pl.plot(clump['t'], clump['Ar']/clump['Ar'][ind2].max(), 'b', alpha=.5)
    pl.plot(clump['t'][ind2], clump['Ar'][ind2]/clump['Ar'][ind2].max(), 'g', alpha=.5)
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
    pl.plot(clump['t'][ind1], clump['fitResults_sigma'][ind1])
    pl.plot(clump['t'][ind2], clump['fitResults_sigmag'][ind2], alpha=.5)
    
    pl.ylim(0, 1e3)
    pl.grid()
    
    pl.ylabel('Fit $\sigma$ [nm]')
    pl.xlabel('Time [frames]')
    pl.legend(['Lipid', 'Cargo'])
    
    pl.subplot(615)
    if plotRaw:
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

def plotEvent3(clump, rawData = None, fitRes = None):
    from scipy import ndimage
    
    dt = 18.27
    pl.figure()
    pl.subplot(211)
    
    c = clump
    if not fitRes is None:
        t0 = fitRes['t0']
    else:
        t0 = 0
        
    i1 = abs(c['fitError_Ag']) < 500
    
    t = c['t'][i1]
    I = c['Ag'][i1]

    t = np.hstack([t[0] - 1, t])
    I = np.hstack([0, I])    
    
    pl.plot((t - t0)*dt, ndimage.median_filter(I, 3), lw=4)
    
    
    pl.plot([0,0], pl.ylim(), 'k:')
    
    pl.yticks([])
    
    pl.xlim(-20*dt, 100*dt)
    
    pl.xticks([])
    #pl.axes()
    pl.box()

    
    pl.subplot(212)
    
    i2 = (abs(c['fitError_Ar']/c['fitResults_Ar']) < 2)*(c['fitResults_sigmag'] > 100)
    pl.plot((c['t'][i2] - t0)*dt, ndimage.median_filter(c['Ar'][i2], 3), 'r', lw=4)
    pl.xlim(-20*dt, 100*dt)
    pl.yticks([])
    pl.plot([0,0], pl.ylim(), 'k:')
    
    ysb = .2*np.mean(pl.ylim())
    pl.plot([1500, 1700], [ysb, ysb], 'k', lw=8)
    pl.text(1500, ysb + .6*ysb, '200 ms')
    pl.xticks([])
    #pl.axes()
    pl.box()
    
    pl.figure()
    
    tvals = np.array([-20, -5, 1, 5, 10, 20, 50, 100]) + int(t0)
    vs = clump.pipeline.mdh.voxelsize_nm.x
    xp = int(clump['x'].mean()/vs)
    yp = int(clump['y'].mean()/vs)
    
    xp1 = int((clump['x'].mean() - clump.pipeline.mdh['chroma.dx'](0,0))/vs)
    yp1 = int((clump['y'].mean() - clump.pipeline.mdh['chroma.dy'](0,0))/vs) + 256
    

    lMax = 1.0*rawData[max(xp-10, 0):(xp+10), max(yp-10, 0):(yp + 10), tvals[3]].max()
    lMin = 1.0*rawData[max(xp-10, 0):(xp+10), max(yp-10, 0):(yp + 10), tvals[0]].min()

    cMax = 1.0*rawData[max(xp1-10, 0):(xp1+10), max(yp1-10, 0):(yp1 + 10), tvals[4]].max()
    cMin = 1.0*rawData[max(xp1-10, 0):(xp1+10), max(yp1-10, 0):(yp1 + 10), tvals[0]].min()     
         
    
    for i, t_i in enumerate(tvals):
        pl.subplot(2,8,i+1)
        frame = rawData[max(xp-10, 0):(xp+10), max(yp-10, 0):(yp + 10), t_i].squeeze()        
        #pl.imshow(frame, interpolation='nearest', cmap='hot', clim=(lMin, lMax))
        
        fr = 1.0 - np.clip(((frame - lMin)/float(lMax - lMin)), 0, 1)[:,:,None]*np.array([1, 1, .5])[None, None, :]
        pl.imshow(fr, interpolation='nearest')
        pl.xticks([])
        pl.yticks([])
        
        pl.subplot(2,8,i+8 + 1)
        frame = rawData[max(xp1-10, 0):(xp1+10), max(yp1-10, 0):(yp1 + 10), t_i].squeeze()        
        #pl.imshow(frame, interpolation='nearest', cmap='hot', clim=(cMin, cMax))
        fr = 1.0 - np.clip(((frame - cMin)/float(cMax - cMin)), 0, 1)[:,:,None]*np.array([.5, 1, 1])[None, None, :]
        pl.imshow(fr, interpolation='nearest')
        pl.xticks([])
        pl.yticks([])
        
    pl.tight_layout(pad=.5)
        

from PYME.Analysis._fithelpers import FitModel, FitModelWeighted

def widthMod(p, t, w0=225.):
    """model for width of lipid signal used to detect the time of fusion as
    indicated by the onset of the lipid difusion into the membrane
    
    the model takes two variable parameters: 
        t0: the time of fusion
        adiff: a scaling factor proportional to the diffusion constant in the membrane
        
    In addition, there is one fixed parameter:
        w0: the width of a vesicle before fusion. For diffraction limited vesicles,
            this should be constant over all vesicles. In this case, the value was
            determined by visual inspecton of a number of traces.
    """
    t0, adiff = p
    
    return w0 + np.sqrt(np.maximum(adiff*(t-t0), 0))

def lipidMod(p, t, t0):
    """This is the lipid signal model from the paper (Eqn ?). The main difference
    is that I have extended the fitting to just before fusion (in the cases where
    fusion and docking do not occur simultaeneously) to allow better estimation
    of Ifus.

    The model is hence: Ifus if t < t0
                        Ifus*(all the rest) if t >= t0
                        
    The variable parameters are:
        tb: the bleaching lifetime
        trel: release time
        lTIRF: the intensiy factor
        Ifus: the intensity imediately prior to fusion
        
    An aditional fixed parameter:
        t0: the time of fusion (determined by fitting the knee in the vesicle width curve)
    
    """
    tb, trel, lTIRF, Ifus = p
    
    t = t - t0    
    
    return Ifus*((t<0) + (t>=0)*(np.exp(-t/trel) + (np.exp(-t/tb) - np.exp(-t/trel))/(lTIRF*(1-trel/tb))))
    
def cargoMod(p, t, t0, Ib = 0):
    """This is an emperical model for the content release events.
    
    It consists of:
        - A  $1-e^(-t/trel)$ dequenching term
        - A standard $e^-(t/tbl)$ bleaching term
        - And a burst or detachment term. The 'burst' happens at the timepoint tbr
          and follows the same kinetics as the initial dequenching. When fitting, 
          tbr is initialised to be very long, so it should only ever take effect
          when there is actually a 'turn off' signal.
          
    This takes 2 fixed parameters:
        t0: time of fusion
        Ib: background intensity
          
    """
        
    I0, trel, tbl, tbr = p
    
    return Ib + I0*(1-np.exp(-(t-t0)/trel))*(t>t0)*np.exp(-(t-t0)/tbl)*(1-np.exp(-(tbr - t)/trel))*(t < tbr) 
    
def fitIntensities(clump):
    from scipy import ndimage 
    pl.figure(figsize=(9, 9))
    
    pl.subplot(311)
    
    ind1 = (clump['fitError_Ag'] < 1e3)*(clump['fitError_Ag'] > 0)
    ind2 = (clump['fitError_Ar'] < 1e3)*(clump['fitError_Ar'] > 0)*(clump['Ar'] > 100) 
    
    try:    
        t0_guess = np.where(ndimage.median_filter(clump['sig'], 5) > 400)[0][0]
    except IndexError:
        t0_guess = 10
        
    #print t0_guess

    #fit to clump widths to find initial 'knee' - ie fusion time    
    t = clump['t'][ind1][max(t0_guess - 20, 0):(t0_guess + 40)].astype('f')
    w = clump['fitResults_sigma'][ind1][max(t0_guess - 20, 0):(t0_guess + 40)]
    s = clump['fitError_sigma'][ind1][max(t0_guess - 20, 0):(t0_guess + 40)]
    
    pl.plot(clump['t'][ind1], clump['fitResults_sigma'][ind1])
    
    rw = FitModelWeighted(widthMod, [t.min() + 20., 100.], w, s, t)
    #print rw[0]
    
    t0, adiff = rw[0]
    
    t0 = max(t0, t.min())
    pl.plot(t, widthMod(rw[0], t), lw=3)
    pl.ylabel('Lipid width ($\sigma$) [nm]')
    pl.xlabel('Time [frames]')
    pl.grid()
    
    pl.title('clumpIndex=%d' % (clump['clumpIndex'][0], ))    
    
    ###################    
    # Lipid Intensity
    
    pl.subplot(312)
    #pl.plot(clump['t'], clump['Ag']/clump['Ag'][ind1].max(), 'g')

    t = clump['t'][ind1].astype('f')
    #t = t - t.min()
    I = clump['Ag'][ind1]
    s = clump['fitError_Ag']*(clump['fitResults_sigma']**2)*clump.pipeline.mapping.a_norm
    s = s[ind1] 
    
    Inorm = I[((t-t0) < 10)*(t> (t0+2))].mean()
    
    
    #plot raw data
    pl.plot(t, I/Inorm, 'x-')
    
    #select a region and fit to the lipid model
    i3 = ((t-t0) > -10)*((t-t0) < 100)
    
    pfit = FitModelWeighted(lipidMod, [300., 3, .2, I[(t-t0) <=0].mean()], I[i3], s[i3], t[i3], t0)
    #print pfit[0]
    
    #plot fit results
    t_ = np.arange(t[i3].min(), t[i3].max()).astype('f')
    pl.plot(t_, lipidMod(pfit[0], t_, t0)/Inorm, lw=3)
    pl.text((t.mean() + t[-1])/2, .5, '$t_0=%3.1f$\n$\\tau_{bleach}=%3.1f$\n$\\tau_{rel}=%3.2f$\n$\lambda_{TIRF}=%3.2f$' % tuple([t0,] + list(pfit[0][:3])), bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    #pl.text((t.mean() + t[-1])/2, .5, '$I_0=%3.1f$\n$\\tau_{rel}=%3.1f$\n$\\tau_{bl}=%3.2f$\n$t_{burst}=%3.2f$' % tuple(list(pfitC[0])), bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    #pl.plot(t, lipidMod([300., 3, .2, .2*I.max()], t, t.min())/I.max(), 'b')

    pl.ylim(0, 1.3)
    pl.grid()
    
    pl.ylabel('Lipid Intensity [a.u.]')
    pl.xlabel('Time [frames]')    
    
    pl.subplot(313)
    
    ###############
    #Cargo
    
    t = clump['t'][ind2].astype('f')
    #t = t - t.min()
    I = clump['Ar'][ind2]
    s = clump['fitError_Ar']*(clump['fitResults_sigmag']**2)*clump.pipeline.mapping.a_norm
    s = s[ind2]
    
    Inorm = I[((t-t0) < 20)*(t> (t0+5))].mean()
    #Ib = np.median(I[t<t0])
    Ib = min(I)
    
    pl.plot(t, clump['Ar'][ind2]/Inorm, '+-', alpha=.5)
    
    
    #select a region to fit over
    i3 = ((t-t0) > -10)*((t-t0) < 150)
    
    #do the fit
    pfitC = FitModelWeighted(cargoMod, [Inorm, 5., 500., 500.+ t0], I[i3], s[i3], t[i3], t0, Ib)
    
    #print pfitC[0]
    
    #plot the fit results
    t_ = np.arange(t[i3].min(), t[i3].max()).astype('f')
    pl.plot(t_, cargoMod(pfitC[0], t_, t0, Ib)/Inorm, lw=2)
    
    
    pl.text((t.mean() + t[-1])/2, .5, '$I_0=%3.1f$\n$\\tau_{rel}=%3.1f$\n$\\tau_{bl}=%3.2f$\n$t_{burst}=%3.2f$' % tuple(list(pfitC[0])), bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))    
    #pl.plot(t_, cargoMod([Inorm, 5., 500., 50.+ t0], t_, t0, Ib)/Inorm, lw=2)
    
    pl.ylim(0, 1.3)
    pl.grid()
    
    pl.ylabel('Cargo Intensity [a.u.]')
    pl.xlabel('Time [frames]')
    #pl.legend(['Lipid', 'fit', 'Cargo', 'fit'])
    
    ############
    # Package up our fit results
    d = {}
    d['t0'], d['bdiffuse'] = rw[0]
    d['tb_lip'], d['trel_lip'], d['lTIRF_lip'], d['Ifus_lip'] = pfit[0]
    d['I0_cargo'], d['trel_cargo'], d['tbl_cargo'], d['tbr_cargo'] = pfitC[0]
    
    return d
    
   
def prepPipeline(pipeline):
    #we fit peak intensity - define mappings which are propotional to integrated intensity instead
    pipeline.mapping.a_norm = 2*np.pi/(pipeline.mdh.voxelsize_nm.x)**2
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
    

def selectAndPlotEvents(pipeline, outputdir='/Users/david/FusionAnalysis', speckleFile = None):
    import os
    import pandas as pd
    from PYME.IO.FileUtils.readSpeckle import readSpeckles    
    #now iterate through our clumps
    clumpIndices = list(set(pipeline['clumpIndex']))
    
    clumps = [pipeline.clumps[i] for i in clumpIndices]
    
    
    
    if not speckleFile is None: #use speckle file to determine which tracks correspond to fusion events
        vs = pipeline.mdh.voxelsize_nm.x
        speckles = readSpeckles(speckleFile)
        
        #print speckles
        
        sp = np.array([s[0,:] for s in speckles])
        #print sp.shape
        
        #filteredClumps = [c for c in clumps if (((c['x'][0] - vs*sp[:,1])**2 + (c['y'][0] - vs*sp[:,0])**2 + (5*(c['t'][0] - sp[:,2]))**2).min() < 300**2)]
        
        #find those clumps which are near (< 1um) to events identified in Joergs speckle file        
        filteredClumps = [c for c in clumps if (((c['x'][0] - vs*sp[:,1])**2 + (c['y'][0] - vs*sp[:,0])**2).min() < 1000**2)]
    
    else:
        #do another level of filtering - fusion events expand, so we're looking for larger than normal
        #sigma in the lipid channel. We can also add a constraint on the mean intensity as proper docking
        #and fusion events are brighter than a lot of the point-like rubbish
        #We do this here, so we can filter on the aggregate behaviour of a track and be more resiliant against
        #noise.
        
        filteredClumps = [c for c in clumps if (c['Ag'].mean() > 2000) and (c['fitResults_sigma'].mean() > 300)]
            
    outputDir = os.path.join(outputdir, os.path.split(pipeline.filename)[1])
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    for c in filteredClumps:
        plotEvent(c, pipeline, plotRaw=True)
        pl.savefig('%s/track%d.pdf' % (outputDir, c['clumpIndex'][0]))
        
        r = fitIntensities(c)
        pl.savefig('%s/track%d_fits.pdf' % (outputDir, c['clumpIndex'][0]))
        
        r['filename'] = os.path.split(pipeline.filename)[1]
        r['clumpIndex'] = c['clumpIndex'][0]
        
        d = {}
        d.update(c)
        pd.DataFrame(d).to_csv('%s/track%d.csv' % (outputDir, c['clumpIndex'][0]))
        #pd.DataFrame(r).to_csv('%s/track%d_fitResults.csv' % (outputDir, c['clumpIndex'][0]))
        with open('%s/track%d_fitResults.json' % (outputDir, c['clumpIndex'][0]), 'w') as f:        
            json.dump(r, f)
        
    return filteredClumps
    
    
def main():
    import sys
    from PYME.LMVis import pipeline
    from PYME.IO.image import ImageStack     
    
    resultFile, imageFile, speckles = sys.argv[1:]
    
    pipe = pipeline.Pipeline()
    pipe.OpenFile(resultFile)
    
    prepPipeline(pipe)
    
    img = image.ImageStack(filename=imageFile, mdh='/Users/david/Downloads/JN150629c3_4_MMStack_Pos0.ome.md')
    selectAndPlotEvents(pipe, speckleFile = speckles)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
