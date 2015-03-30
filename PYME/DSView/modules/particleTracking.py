# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:15:36 2015

@author: david
"""

import wx
import PYME.misc.autoFoldPanel as afp
import numpy as np
import pylab

from PYME.Analysis.Tracking import tracking

from traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, ListInstance, on_trait_change
from traitsui.api import View, Item, Group

class ParticleTracker(HasTraits):
    features = CStr('x, y')    
    pNew = Float(0.2)
    r0 = Float(500)
    pLinkCutoff = Float(0.2)
    
    showTracks = Bool(True)
    showCandidates = Bool(True)
    
    candLineWidth = Int(4)
    chosenLineWidth = Int(5)
    trackLineWidth = Int(2)
    
    traits_view = View(Group(Item(name = 'features'),
                             Item(name = 'pNew'),
                             Item(name = 'r0'),
                             Item(name = 'pLinkCutoff')),
                       Group(Item(name = 'showTracks'),
                             Item(name = 'showCandidates')))
    
    def __init__(self, dsviewer):
        HasTraits.__init__(self)
        self.dsviewer = dsviewer
        self.view = dsviewer.view
        self.do = dsviewer.do
        self.image = dsviewer.image
        
        self.tracker = None
        
        
#        self.features.on_trait_change(self.OnFeaturesChanged)
#        self.pNew.on_trait_change(self.OnParamChange)
#        self.r0.on_trait_change = self.OnParamChange
#        self.pLinkCutoff.on_trait_change = self.OnParamChange
        
        #self.pipeline = dsviewer.pipeline
        self.penCols = [wx.Colour(*pylab.cm.hsv(v, bytes=True)) for v in np.linspace(0, 1, 16)]
        self.penColsA = [wx.Colour(*pylab.cm.hsv(v, alpha=0.5, bytes=True)) for v in np.linspace(0, 1, 16)]
        self.CreatePens()
        
        dsviewer.do.overlays.append(self.DrawOverlays)

        dsviewer.paneHooks.append(self.GenTrackingPanel)
    
    @on_trait_change('candLineWidth, chosenLineWidth, trackLineWidth')    
    def CreatePens(self):
        self.candPens = [wx.Pen(c, self.candLineWidth, wx.DOT) for c in self.penCols]
        self.chosenPens = [wx.Pen(c, self.chosenLineWidth) for c in self.penCols]
        self.trackPens = [wx.Pen(c, self.trackLineWidth) for c in self.penColsA]

    def GenTrackingPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Particle Tracking", pinned = True)
        
        pan = self.edit_traits(parent=item, kind='panel')
        item.AddNewElement(pan.control)
        
        #pan = wx.Panel(item, -1)

#        vsizer = wx.BoxSizer(wx.VERTICAL)
#
##        #if self.multiChannel: #we have channels            
#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#        hsizer.Add(wx.StaticText(pan, -1, 'Features:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#    
#        self.tFeatures = wx.Text(pan, -1, 'x, y')
#            
#        hsizer.Add(self.tFeatures, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#    
#        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)        
#        
#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#
#
#        
#        vsizer.Add(hsizer, 0,wx.ALL|wx.ALIGN_RIGHT, 5)
#        
#        
#        pan.SetSizer(vsizer)
#        vsizer.Fit(pan)


        #item.AddNewElement(pan)
        bTrack = wx.Button(item, -1, 'Track')
        bTrack.Bind(wx.EVT_BUTTON, self.OnTrack)
        item.AddNewElement(bTrack)
        
        _pnl.AddPane(item)

    
    @on_trait_change('pNew, r0, pLinkCutoff')    
    def OnParamChange(self):
        if not self.tracker == None:
            self.tracker.pNew=self.pNew
            self.tracker.r0 = self.r0
            self.tracker.linkageCuttoffProb = self.pLinkCutoff
            
    @on_trait_change('features')   
    def OnFeaturesChanged(self):
        self.tracker = None

    def OnTrack(self, event):
        pipeline = self.dsviewer.pipeline
        
        if self.tracker == None:
            featNames = [s.strip() for s in self.features.split(',')]
            
            def _calcWeights(s):
                fw = s.split('*')
                if len(fw) == 2:
                    return float(fw[0]), fw[1]
                else:
                    return 1.0, s
                    
            
            weightedFeats = [_calcWeights(s) for s in featNames]
            
            feats = np.vstack([w*pipeline[fn] for w, fn in weightedFeats])
            
            self.tracker = tracking.Tracker(pipeline['t'], feats)
            
            self.tracker.pNew=self.pNew
            self.tracker.r0 = self.r0
            self.tracker.linkageCuttoffProb = self.pLinkCutoff

        for i in range(1, self.dsviewer.image.data.shape[2]):
            L = self.tracker.calcLinkages(i,i-1)
            self.tracker.updateTrack(i, L)
        
        pipeline.selectedDataSource.clumps = self.tracker.clumpIndex
        pipeline.selectedDataSource.setMapping('clumpIndex', 'clumps')
        
        clumpSizes = np.zeros_like(self.tracker.clumpIndex)
        
        for i in set(self.tracker.clumpIndex):
            ind = (self.tracker.clumpIndex == i)
            
            clumpSizes[ind] = ind.sum()
            
        pipeline.selectedDataSource.clumpSizes = clumpSizes
        pipeline.selectedDataSource.setMapping('clumpSize', 'clumpSizes')
            
        

        
    def DrawOverlays(self, view, dc):
        if self.showTracks and not (self.tracker == None):
            t = self.dsviewer.pipeline['t']
            x = self.dsviewer.pipeline['x']/self.image.voxelsize[0]
            y = self.dsviewer.pipeline['y']/self.image.voxelsize[1]
            
            xb, yb, zb = view._calcVisibleBounds()
            
            IFoc = (x >= xb[0])*(y >= yb[0])*(t >= (zb[0]-5))*(x < xb[1])*(y < yb[1])*(t < (zb[1]+5))
            
            tFoc = list(set(self.tracker.clumpIndex[IFoc]))

            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            #pGreen = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            #pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            #dc.SetPen(pGreen)
            

            for tN in tFoc:
                IFoc = (self.tracker.clumpIndex == tN)
                if IFoc.sum() >= 2:
                    pFoc = np.vstack(view._PixelToScreenCoordinates3D(x[IFoc], y[IFoc], t[IFoc])).T
                    
                    #print pFoc.shape
                    dc.SetPen(self.trackPens[tN%16])
                    dc.DrawSpline(pFoc)
                
        if self.showCandidates and not (self.tracker == None):
            if view.do.zp >=1:
                iCurr = view.do.zp
                iPrev = view.do.zp-1
                links = self.tracker.calcLinkages(iCurr, iPrev)
                
                #pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),2)
                pRedDash = wx.Pen(wx.TheColourDatabase.FindColour('RED'),2, wx.SHORT_DASH)
                dc.SetPen(pRedDash)
                
                dc.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))                
                dc.SetTextForeground(wx.TheColourDatabase.FindColour('YELLOW'))
                
                for curFrameIndex, linkInfo in links.items():
                    inds = self.tracker.indicesByT[iCurr]
                    i = inds[curFrameIndex]
                    
                    x1 = self.dsviewer.pipeline['x'][i]/self.image.voxelsize[0] 
                    y1 = self.dsviewer.pipeline['y'][i]/self.image.voxelsize[1]
                    
                    x1s, y1s = view._PixelToScreenCoordinates(x1, y1)
                    
                    linkSrcs, linkPs = linkInfo
                    n = 0
                    for ls, lp in zip(linkSrcs, linkPs):
                        if n == 0:
                            dc.SetPen(self.chosenPens[self.tracker.clumpIndex[ls]%16])
                        else:
                            dc.SetPen(self.candPens[self.tracker.clumpIndex[ls]%16])
                            
                        if ls == -1:
                            #new object
                            x0 = x1
                            y0 = y1 - 10
                            
                        else:
                            x0 = self.dsviewer.pipeline['x'][ls]/self.image.voxelsize[0]
                            y0 = self.dsviewer.pipeline['y'][ls]/self.image.voxelsize[1]
                        
                        x0s, y0s = view._PixelToScreenCoordinates(x0, y0)
                        dc.DrawLine(x0s, y0s, x1s, y1s)
                        
                        if ls == -1:
                            dc.DrawText('N', x0s, y0s + 1)
                        dc.DrawText('%1.1f'%lp, (x0s+x1s)/2 + 2, (y0s+y1s)/2 + 2)
                        n += 1
                    
                    
                    
                    
                    

        
  

def Plug(dsviewer):
    dsviewer.tracker = ParticleTracker(dsviewer)