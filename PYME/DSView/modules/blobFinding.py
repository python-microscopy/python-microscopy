#!/usr/bin/python
##################
# blobFinding.py
#
# Copyright David Baddeley, 2011
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

import wx
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
from PYME.ui import recArrayView
import numpy
from PYME.DSView.OverlaysPanel import OverlayPanel
import wx.lib.agw.aui as aui
import os
import six


from ._base import Plugin

class BlobFinder(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)

        self.vObjPos = None
        self.vObjFit = None
        self.nObjFit = 0
        
        dsviewer.AddMenuItem("Save", "Save &Positions", self.savePositions)
        dsviewer.AddMenuItem("Save", "Save &Fit Results", self.saveFits)
        dsviewer.AddMenuItem("Save", "Save shift maps", self.saveShiftmaps)

        dsviewer.paneHooks.append(self.GenBlobFindingPanel)
        dsviewer.paneHooks.append(self.GenBlobFitPanel)

    def GenBlobFindingPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Object Finding", pinned = True)

        pan = wx.Panel(item, -1)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Threshold:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tThreshold = wx.TextCtrl(pan, -1, value='50', size=(40, -1))

        hsizer.Add(self.tThreshold, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        pan.SetSizer(hsizer)
        hsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        item.AddNewElement(pan)
        
        pan = wx.Panel(item, -1)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        #hsizer.Add(wx.StaticText(pan, -1, 'Method:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.chMethod = wx.Choice(pan, -1, choices=['Simple Threshold', 'SNR Threshold', 'Multi-threshold'])
        self.chMethod.SetSelection(0)
        self.chMethod.Bind(wx.EVT_CHOICE, self.OnChangeMethod)

        hsizer.Add(self.chMethod, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        

        pan.SetSizer(hsizer)
        hsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        item.AddNewElement(pan)
        
        pan = wx.Panel(item, -1)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Channel:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
    
        self.chChannel = wx.Choice(pan, -1, choices=self.do.names)
            
        hsizer.Add(self.chChannel, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
    
        pan.SetSizer(hsizer)
        hsizer.Fit(pan) 
        
        item.AddNewElement(pan)
        
        pan = wx.Panel(item, -1)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Blur size:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tBlurSize = wx.TextCtrl(pan, -1, value='1.5', size=(40, -1))
        self.tBlurSize.Disable()

        hsizer.Add(self.tBlurSize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        pan.SetSizer(hsizer)
        hsizer.Fit(pan)

        #_pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        item.AddNewElement(pan)

        #self.cbSNThreshold = wx.CheckBox(item, -1, 'SNR Threshold')
        #self.cbSNThreshold.SetValue(False)

        #_pnl.AddFoldPanelWindow(item, self.cbSNThreshold, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        #item.AddNewElement(self.cbSNThreshold)
        
        #self.cbThresholdRange = wx.CheckBox(item, -1, 'Multi-thresholds')
        #self.cbThresholdRange.SetValue(False)

        #_pnl.AddFoldPanelWindow(item, self.cbSNThreshold, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        #item.AddNewElement(self.cbThresholdRange)

        bFindObjects = wx.Button(item, -1, 'Find')


        bFindObjects.Bind(wx.EVT_BUTTON, self.OnFindObjects)
        #_pnl.AddFoldPanelWindow(item, bFindObjects, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        item.AddNewElement(bFindObjects)
        _pnl.AddPane(item)
        
    def OnChangeMethod(self, event):
        thresholds = [50, 1, 1]
        sel = self.chMethod.GetSelection()
        
        if sel == 2: #multi-threshold
            self.tBlurSize.Enable()
            #self.tThreshold.SetValue('4')
        else:
            self.tBlurSize.Disable()
        
        #if sel == 1:
        self.tThreshold.SetValue('%3.1f' % thresholds[sel])
            
        #if sel == 

    def OnFindObjects(self, event):
        threshold = float(self.tThreshold.GetValue())
        chnum = self.chChannel.GetSelection()

        from PYME.localization.ofind3d import ObjectIdentifier

        if not 'ofd' in dir(self) or not self.ofd.chnum == chnum:
            #create an object identifier
            self.ofd = ObjectIdentifier(self.image.data[:,:,:, chnum])
            self.ofd.chnum = chnum

        #and identify objects ...
        if self.chMethod.GetSelection() == 1: #don't detect objects in poisson noise
            fudgeFactor = 1 #to account for the fact that the blurring etc... in ofind doesn't preserve intensities - at the moment completely arbitrary so a threshold setting of 1 results in reasonable detection.
            threshold =  (numpy.sqrt(self.image.mdh.Camera.ReadNoise**2 + numpy.maximum(self.image.mdh.Camera.ElectronsPerCount*(self.image.mdh.Camera.NoiseFactor**2)*(self.image.data[:,:,:, chnum].astype('f4') - self.image.mdh.Camera.ADOffset)*self.image.mdh.Camera.TrueEMGain, 1))/self.image.mdh.Camera.ElectronsPerCount)*fudgeFactor*threshold
            self.ofd.FindObjects(threshold, 0)
        elif self.chMethod.GetSelection() == 2:
            bs = float(self.tBlurSize.GetValue())
            self.ofd.FindObjects(threshold, blurRadius = bs, blurRadiusZ = bs)
        else:
            self.ofd.FindObjects(threshold,0)

        self.dsviewer.view.points = numpy.array([[p.x, p.y, p.z] for p in self.ofd])

        self.objPosRA = numpy.rec.fromrecords(self.dsviewer.view.points, names='x,y,z')

        if self.vObjPos is None:
            self.vObjPos = recArrayView.ArrayPanel(self.dsviewer, self.objPosRA)
            self.dsviewer.AddPage(self.vObjPos, caption='Object Positions')
        else:
            self.vObjPos.grid.SetData(self.objPosRA)

        self.dsviewer.update()

    def GenBlobFitPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Object Fitting", pinned = True)
#        item = _pnl.AddFoldPanel("Object Fitting", collapsed=False,
#                                      foldIcons=self.Images)

        bFitObjects = wx.Button(item, -1, 'Fit')


        bFitObjects.Bind(wx.EVT_BUTTON, self.OnFitObjects)
        #_pnl.AddFoldPanelWindow(item, bFitObjects, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        item.AddNewElement(bFitObjects)

        pan = wx.Panel(item, -1)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Bead wxy:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tBeadWXY = wx.TextCtrl(pan, -1, value='125', size=(40, -1))

        hsizer.Add(self.tBeadWXY, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        pan.SetSizer(hsizer)
        hsizer.Fit(pan)

        item.AddNewElement(pan)
        
        bCalcShiftMap = wx.Button(item, -1, 'Shiftmap')
        bCalcShiftMap.Bind(wx.EVT_BUTTON, self.OnCalcShiftmap)
        #_pnl.AddFoldPanelWindow(item, bFitObjects, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        item.AddNewElement(bCalcShiftMap)
        _pnl.AddPane(item)

    def OnFitObjects(self, event):
        import PYME.localization.FitFactories.Gauss3DFitR as fitMod
        from PYME.IO import MetaDataHandler
        chnum = self.chChannel.GetSelection()
        
        mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
        mdh['tIndex'] = 0
        
        self.objFitRes = {}
        
        for chnum in range(self.image.data.shape[3]):
            data = self.image.data[:,:,:,chnum]  
            
            if not 'Camera.ADOffset' in mdh.getEntryNames():
                mdh['Camera.ADOffset'] = data.min()
    
            fitFac = fitMod.FitFactory(data, mdh)
    
            self.objFitRes[chnum] = numpy.empty(len(self.ofd), fitMod.FitResultsDType)
            for i in range(len(self.ofd)):
                p = self.ofd[i]
                #try:
                self.objFitRes[chnum][i] = fitFac.FromPoint(round(p.x), round(p.y), round(p.z), 8)
                #except:
                #    pass
    
    
            #if self.nObjFit == None:
                
                
            vObjFit = recArrayView.ArrayPanel(self.dsviewer, self.objFitRes[chnum]['fitResults'])
            self.dsviewer.AddPage(vObjFit, caption = 'Fitted Positions %d - %d' % (chnum, self.nObjFit))
        self.nObjFit += 1
        #else:
        #    self.vObjFit.grid.SetData(self.objFitRes['fitResults'])

        self.dsviewer.update()
        
    def OnCalcShiftmap(self, event):
        from PYME.Analysis.points import twoColour, twoColourPlot
        # import pylab
        import matplotlib.pyplot as plt
        masterChan = self.chChannel.GetSelection()
        
        master = self.objFitRes[masterChan]
        x0 = master['fitResults']['x0']
        y0 = master['fitResults']['y0']
        err_x0 = master['fitError']['x0']
        err_y0 = master['fitError']['y0']
        z0 = master['fitResults']['z0']

        wxy = master['fitResults']['wxy']

        wxy_bead = float(self.tBeadWXY.GetValue())

        mask = numpy.abs(wxy - wxy_bead) < (.25*wxy_bead)
        
        self.shiftfields ={}

        plt.figure() 
        
        nchans = self.image.data.shape[3]
        ch_i = 1
        
        for ch in range(nchans):
            if not ch == masterChan:
                res = self.objFitRes[ch]
                
                x = res['fitResults']['x0']
                y = res['fitResults']['y0']
                z = res['fitResults']['z0']
                err_x = numpy.sqrt(res['fitError']['x0']**2 + err_x0**2)
                err_y = numpy.sqrt(res['fitError']['y0']**2 + err_y0**2)
                
                dx = x - x0
                dy = y - y0
                dz = z - z0
                
                print(('dz:', numpy.median(dz[mask])))
                
                
                spx, spy = twoColour.genShiftVectorFieldLinear(x[mask], y[mask], dx[mask], dy[mask], err_x[mask], err_y[mask])
                self.shiftfields[ch] = (spx, spy, numpy.median(dz[mask]))
                #twoColourPlot.PlotShiftField2(spx, spy, self.image.data.shape[:2])
                
                plt.subplot(1,nchans -1, ch_i)
                ch_i += 1
                twoColourPlot.PlotShiftResidualsS(x[mask], y[mask], dx[mask], dy[mask], spx, spy)
                
        plt.figure()
        X, Y = numpy.meshgrid(numpy.linspace(0., 70.*self.image.data.shape[0], 20), numpy.linspace(0., 70.*self.image.data.shape[1], 20))
        X = X.ravel()
        Y = Y.ravel()
        for k in self.shiftfields.keys():
            spx, spy, dz = self.shiftfields[k]
            plt.quiver(X, Y, spx.ev(X, Y), spy.ev(X, Y), color=['r', 'g', 'b'][k], scale=2e3)
            
        plt.axis('equal')
        
    def saveShiftmaps(self, event=None):
        from six.moves import cPickle
        for k in self.shiftfields.keys():
            fdialog = wx.FileDialog(None, 'Save Positions ...',
                wildcard='Shiftmap|*.sm', defaultFile=os.path.splitext(self.image.names[k])[0] + '.sm', style=wx.FD_SAVE)
            succ = fdialog.ShowModal()
            if (succ == wx.ID_OK):
                outFilename = fdialog.GetPath()
                
                fid = open(outFilename, 'wb')
                cPickle.dump(self.shiftfields[k], fid, 2)
                fid.close()
                

    def savePositions(self, event=None):
        fdialog = wx.FileDialog(None, 'Save Positions ...',
            wildcard='Tab formatted text|*.txt', defaultFile=os.path.splitext(self.image.seriesName)[0] + '_pos.txt', style=wx.FD_SAVE)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath()

            of = open(outFilename, 'w')
            of.write('\t'.join(self.objPosRA.dtype.names) + '\n')

            for obj in self.objPosRA:
                of.write('\t'.join([repr(v) for v in obj]) + '\n')
            of.close()

            npFN = os.path.splitext(outFilename)[0] + '.npy'

            numpy.save(npFN, self.objPosRA)

    def saveFits(self, event=None):
        fdialog = wx.FileDialog(None, 'Save Fit Results ...',
            wildcard='Tab formatted text|*.txt', defaultFile=os.path.splitext(self.image.seriesName)[0] + '_fits.txt', style=wx.FD_SAVE)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath()

            of = open(outFilename, 'w')
            of.write('\t'.join(self.objFitRes['fitResults'].dtype.names) + '\n')

            for obj in self.objFitRes['fitResults']:
                of.write('\t'.join([repr(v) for v in obj]) + '\n')
            of.close()

            npFN = os.path.splitext(outFilename)[0] + '.npy'

            numpy.save(npFN, self.objFitRes)


def Plug(dsviewer):
    blobFinder = BlobFinder(dsviewer)
    
    if not 'overlaypanel' in dir(dsviewer):    
        dsviewer.overlaypanel = OverlayPanel(dsviewer, dsviewer.view, dsviewer.image.mdh)
        dsviewer.overlaypanel.SetSize(dsviewer.overlaypanel.GetBestSize())
        pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(dsviewer.overlaypanel, pinfo2)
    
        dsviewer.panesToMinimise.append(pinfo2)
        
    return blobFinder
    
