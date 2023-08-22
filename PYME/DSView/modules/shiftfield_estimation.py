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
from PYME.ui import progress
import numpy
import numpy as np
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
        
        dsviewer.AddMenuItem("File>Save", "Save &Positions", self.savePositions)
        dsviewer.AddMenuItem("File>Save", "Save &Fit Results", self.saveFits)
        dsviewer.AddMenuItem("File>Save", "Save shift maps", self.saveShiftmaps)

        dsviewer.paneHooks.append(self.GenBlobFindingPanel)
        #dsviewer.paneHooks.append(self.GenBlobFitPanel)
        dsviewer.paneHooks.append(self.GenShiftMapPanel)

    def GenBlobFindingPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Object Finding", pinned = True)
        
        pan = wx.Panel(item, -1)
        vsizer=wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Threshold:'), 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tThreshold = wx.TextCtrl(pan, -1, value='50', size=(40, -1))
        hsizer.Add(self.tThreshold, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.chMethod = wx.Choice(pan, -1, choices=['Simple Threshold', 'SNR Threshold', 'Multi-threshold'])
        self.chMethod.SetSelection(0)
        self.chMethod.Bind(wx.EVT_CHOICE, self.OnChangeMethod)

        hsizer.Add(self.chMethod, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.ALL, 5)
        
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer.Add(wx.StaticText(pan, -1, 'Channel:'), 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
    
        # self.chChannel = wx.Choice(pan, -1, choices=self.do.names)
            
        # hsizer.Add(self.chChannel, 1,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        # vsizer.Add(hsizer, 0, wx.ALL, 5)


        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(pan, -1, 'Blur size:'), 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tBlurSize = wx.TextCtrl(pan, -1, value='1.5', size=(40, -1))
        self.tBlurSize.Disable()

        hsizer.Add(self.tBlurSize, 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.ALL, 5)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

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

        #bFindObjects = wx.Button(item, -1, 'Find')


        #bFindObjects.Bind(wx.EVT_BUTTON, self.OnFindObjects)
        #_pnl.AddFoldPanelWindow(item, bFindObjects, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        #item.AddNewElement(bFindObjects)
    #     _pnl.AddPane(item)
        
    



    # def GenBlobFitPanel(self, _pnl):
    #     item = afp.foldingPane(_pnl, -1, caption="Object Fitting", pinned = True)
#        item = _pnl.AddFoldPanel("Object Fitting", collapsed=False,
#                                      foldIcons=self.Images)

        pan = wx.Panel(item, -1)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'ROI half size:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tROIsize = wx.TextCtrl(pan, -1, value='8', size=(40, -1))
        hsizer.Add(self.tROIsize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        pan.SetSizer(hsizer)
        hsizer.Fit(pan)
        item.AddNewElement(pan)
        
        bFitObjects = wx.Button(item, -1, 'Find and Fit')
        bFitObjects.Bind(wx.EVT_BUTTON, self.OnFitObjects)
        #_pnl.AddFoldPanelWindow(item, bFitObjects, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        item.AddNewElement(bFitObjects)
        _pnl.AddPane(item)

    def OnChangeMethod(self, event):
        thresholds = [50, 1, 1]
        sel = self.chMethod.GetSelection()
        
        if sel == 2: #multi-threshold
            self.tBlurSize.Enable()
            #self.tThreshold.SetValue('4')
        else:
            self.tBlurSize.Disable()

        self.tThreshold.SetValue('%3.1f' % thresholds[sel])
            


    def OnFitObjects(self, event):
        from PYME.localization.ofind3d import ObjectIdentifier
        import PYME.localization.FitFactories.Gauss3DFitR as fitMod
        from PYME.IO import MetaDataHandler
        from PYME.IO import tabular
        
        mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
        mdh['tIndex'] = 0
        #print(mdh)
        
        self.objFitRes = {}

        thresholds = [float(t) for t in self.tThreshold.GetValue().split(',')]
        if len(thresholds) == 1:
            thresholds = thresholds*self.image.data_xyztc.shape[4]
        
        for chnum in range(self.image.data.shape[3]):
            data = self.image.data[:,:,:,chnum]
            ofd =  ObjectIdentifier(np.atleast_3d(data))

            threshold = thresholds[chnum]

            if self.chMethod.GetSelection() == 1: #don't detect objects in poisson noise
                fudgeFactor = 1 #to account for the fact that the blurring etc... in ofind doesn't preserve intensities - at the moment completely arbitrary so a threshold setting of 1 results in reasonable detection.
                threshold =  (numpy.sqrt(self.image.mdh.Camera.ReadNoise**2 + numpy.maximum(self.image.mdh.Camera.ElectronsPerCount*(self.image.mdh.Camera.NoiseFactor**2)*(self.image.data[:,:,:, chnum].astype('f4') - self.image.mdh.Camera.ADOffset)*self.image.mdh.Camera.TrueEMGain, 1))/self.image.mdh.Camera.ElectronsPerCount)*fudgeFactor*threshold
                ofd.FindObjects(threshold, 0)
            elif self.chMethod.GetSelection() == 2:
                bs = float(self.tBlurSize.GetValue())
                ofd.FindObjects(threshold, blurRadius = bs, blurRadiusZ = bs)
            else:
                ofd.FindObjects(threshold,0)
            
            if not 'Camera.ADOffset' in mdh.getEntryNames():
                mdh['Camera.ADOffset'] = data.min()
    
            fitFac = fitMod.FitFactory(data, mdh)
    
            self.objFitRes[chnum] = numpy.empty(len(ofd), fitMod.FitResultsDType)
            for i in range(len(ofd)):
                p = ofd[i]
                
                self.objFitRes[chnum][i] = fitFac.FromPoint(round(p.x), round(p.y), round(p.z), roiHalfSize=int(self.tROIsize.GetValue()))           
                
            vObjFit = recArrayView.ArrayPanel(self.dsviewer, self.objFitRes[chnum]['fitResults'])
            self.dsviewer.AddPage(vObjFit, caption = 'Fitted Positions %d - %d' % (chnum, self.nObjFit))
        
        self.nObjFit += 1

        f = tabular.ConcatenateFilter(*[tabular.FitResultsSource(self.objFitRes[chnum], sort=False) for chnum in range(self.image.data.shape[3])], concatKey='channel')  

        if not hasattr(self, '_fit_ovl'):
            from PYME.DSView import overlays
            self._fit_ovl = overlays.PointDisplayOverlay(filter=f, display_name='Fitted positions')
            self._fit_ovl.display_as = 'cross'
            self._fit_ovl.z_mode = 'z'
            self.view.add_overlay(self._fit_ovl)
        else:
            self._fit_ovl.filter = f

        self.dsviewer.update()
    
    def GenShiftMapPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Shiftmap", pinned = True)

        pan = wx.Panel(item, -1)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Master Channel:'), 0,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
    
        self.chChannel = wx.Choice(pan, -1, choices=self.do.names)
        self.chChannel.Select(0)
            
        hsizer.Add(self.chChannel, 1,wx.LEFT|wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Bead wxy:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tBeadWXY = wx.TextCtrl(pan, -1, value='125', size=(40, -1))
        hsizer.Add(self.tBeadWXY, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Type:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.cShiftfieldType = wx.Choice(pan, -1, choices=['Linear (shift + rotation)', 'Quadratic (shift, rotation, and field-dependent magnification)', 'Spline interpolation (model free)'], size=(40, -1))
        self.cShiftfieldType.SetSelection(1)
        hsizer.Add(self.cShiftfieldType, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 5)

        pan.SetSizer(vsizer)
        hsizer.Fit(pan)
        item.AddNewElement(pan)
        
        bCalcShiftMap = wx.Button(item, -1, 'Shiftmap')
        bCalcShiftMap.Bind(wx.EVT_BUTTON, progress.managed(self.OnCalcShiftmap, self.dsviewer, 'Calculating shiftmap...'))
        #_pnl.AddFoldPanelWindow(item, bFitObjects, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        item.AddNewElement(bCalcShiftMap)
        _pnl.AddPane(item)


        
    def OnCalcShiftmap(self, event):
        from PYME.Analysis.points import twoColour, twoColourPlot
        from PYME.Analysis import fiducial_matching
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

        mask = np.abs(wxy - wxy_bead) < (.25*wxy_bead)

        pts_master = np.vstack([x0[mask], y0[mask]]).T

        nBeads = mask.sum()
        
        print('Using %d beads for shiftmap' % nBeads)
        
        self.shiftfields ={}

        #plt.figure() 
        
        nchans = self.image.data.shape[3]
        ch_i = 1

        model_type = self.cShiftfieldType.GetSelection()
        
        for ch in range(nchans):
            if not ch == masterChan:
                res = self.objFitRes[ch]
                
                x = res['fitResults']['x0']
                y = res['fitResults']['y0']
                z = res['fitResults']['z0']
                err_x = res['fitError']['x0'] #**2 + err_x0[mask]**2)
                err_y = res['fitError']['y0'] # **2 + err_y0[mask]**2)

                pts = np.vstack([x, y]).T

                i, score = fiducial_matching.match_points(pts_master, pts, 1000.0)

                #TODO - filter on score??

                print('i:', i, score)

                xi, yi, zi, err_xi, err_yi = x[i], y[i], z[i], err_x[i], err_y[i]

                err_x = np.sqrt(err_xi**2 + err_x0[mask]**2)
                err_y = np.sqrt(err_yi**2 + err_y0[mask]**2)
                
                dx = xi - x0[mask]
                dy = yi - y0[mask]
                dz = zi - z0[mask]

                # plt.figure()
                # plt.plot(x0, y0, 'x')
                # plt.plot(x, y, 'o')
                # plt.quiver(x0, y0, dx, dy, color=['r', 'g', 'b'][ch], angles='xy', scale_units='xy', scale=1)

                
                # print(('dz:', numpy.median(dz[mask])))  

                # print(x/1e3, x0/1e3, dx/1e3) 
                
                if model_type == 0:
                    if nBeads < 3:
                        raise ValueError('Not enough beads for linear model (3 d.f), only using %d beads - try adjustin filter wxy' % nBeads)
                    
                    #print(dx[mask]/1e3, dy[mask]/1e3)
                    spx, spy = twoColour.genShiftVectorFieldLinear(xi, yi, dx, dy, err_x, err_y)
                elif model_type == 1:
                    if nBeads < 9:
                        raise ValueError('Not enough beads for quad model (9 d.f.), only using %d beads - try adjustin filter wxy' % nBeads)
                    spx, spy = twoColour.genShiftVectorFieldQuad(xi, yi, dx, dy, err_x, err_y)
                elif model_type == 2:
                    if nBeads < 3:
                        raise ValueError('Not enough beads for spline model, only using %d beads - try adjustin filter wxy' % nBeads)
                    _, _, spx, spy, _ = twoColour.genShiftVectorFieldSpline(xi, yi, dx, dy, err_x, err_y)
                else:
                    raise ValueError('Unknown model type')

                # plt.figure()
                # plt.plot(x0[mask], y0[mask], 'x')
                # plt.plot(x, y, 'o')
                # plt.quiver(x0[mask], y0[mask], dx, dy, np.log10(score), angles='xy', scale_units='xy', scale=1, width=0.01)
                # plt.colorbar()

                self.shiftfields[ch] = (spx, spy, numpy.median(dz))
                #twoColourPlot.PlotShiftField2(spx, spy, self.image.data.shape[:2])
                
                plt.figure()
                plt.subplot(1,nchans -1, ch_i)
                ch_i += 1
                twoColourPlot.PlotShiftResidualsS(xi, yi, dx, dy, spx, spy)
                
        plt.figure()
        X, Y = numpy.meshgrid(numpy.linspace(0., self.image.voxelsize_nm.x*self.image.data.shape[0], 20), numpy.linspace(0., self.image.voxelsize_nm.y*self.image.data.shape[1], 20))
        X = X.ravel()
        Y = Y.ravel()
        for k in self.shiftfields.keys():
            spx, spy, dz = self.shiftfields[k]
            plt.quiver(X, Y, spx.ev(X, Y), spy.ev(X, Y), color=['r', 'g', 'b'][k], angles='xy', scale_units='xy', scale=1)
            
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
    
