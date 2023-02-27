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

import os
import time

import numpy
import numpy as np
import wx
import wx.lib.agw.aui as aui
from PYME.DSView import View3D, ViewIm3D, ImageStack
from PYME.IO import MetaDataHandler
from PYME.ui.mytimer import mytimer
from scipy import ndimage


def _pt(sl):
    dec, psf, d, regLambda, nIter, weights = sl
    dec.psf_calc(psf, d.shape)
    r = dec.deconv(d,regLambda, nIter, weights).reshape(dec.shape)
    #r = 0
    return r

from ._base import Plugin
class Deconvolver(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        self.tq = None
        
        dsviewer.AddMenuItem("Processing", "Deconvolution", self.OnDeconvICTM)
        dsviewer.AddMenuItem("Processing", "Deconvole bead shape", self.OnDeconvBead)
        dsviewer.AddMenuItem("Processing", "Weiner Deconvolution", self.OnDeconvWiener)
        dsviewer.AddMenuItem("Processing", "Deconvolve movie (2D)", self.OnDeconvMovie)

        dsviewer.updateHooks.append(self.update)
        
    def checkTQ(self):
        import Pyro.core
        if self.tq is None:
            #if 'PYME_TASKQUEUENAME' in os.environ.keys():
            #    taskQueueName = os.environ['PYME_TASKQUEUENAME']
            #else:
            #    taskQueueName = 'taskQueue'

            from PYME.misc.computerName import GetComputerName
            compName = GetComputerName()

            taskQueueName = 'TaskQueues.%s' % compName

            self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)


#    def checkTQ(self):
#        import Pyro.core
#        if self.tq == None:
#            if 'PYME_TASKQUEUENAME' in os.environ.keys():
#                taskQueueName = os.environ['PYME_TASKQUEUENAME']
#            else:
#                taskQueueName = 'taskQueue'
#            self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)

    def OnDeconvBead(self, event):
        self.OnDeconvICTM(None, True)

    
    def OnDeconvICTM(self, event, beadMode=False):
        from PYME.Deconv.deconvDialogs import DeconvSettingsDialog, DeconvProgressPanel

        dlg = DeconvSettingsDialog(self.dsviewer, beadMode, self.image.data.shape[3])
        if dlg.ShowModal() == wx.ID_OK:
            from PYME.Deconv import dec, decThread, richardsonLucy
            nIter = dlg.GetNumIterationss()
            regLambda = dlg.GetRegularisationLambda()

            decMDH = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
            decMDH['Deconvolution.NumIterations'] = nIter
            decMDH['Deconvolution.OriginalFile'] = self.image.filename

            vx, vy, vz = self.image.voxelsize_nm

            if beadMode:
                from PYME.Deconv import beadGen
                psf = beadGen.genBeadImage(dlg.GetBeadRadius(), (vx, vy, vz))

                decMDH['Deconvolution.BeadRadius'] = dlg.GetBeadRadius()
                
            else:
                psfFilename, psf, vs = dlg.GetPSF(vshint = vx)
                
                if psf.shape[2] < 2:
                    raise RuntimeError('Expepected a 3D PSF for 3D deconvolution. For 2D deconvolution use the DeconvMovie function')

                decMDH['Deconvolution.PSFFile'] = dlg.GetPSFFilename()

                if not (vs.x == vx and vs.y == vy and vs.z ==vz):
                    #rescale psf to match data voxel size
                    psf = ndimage.zoom(psf, [vs.x/vx, vs.y/vy, vs.z/vz])

            #crop PSF in z if bigger than stack
            if psf.shape[2] > self.image.data.shape[2]:
                dz = psf.shape[2] - self.image.data.shape[2]

                psf = psf[:, :, int(numpy.floor(dz / 2)):int(psf.shape[2] - numpy.ceil(dz / 2))]


            decMDH['Deconvolution.Offset'] = dlg.GetOffset()

            bg = dlg.GetBackground()
            decMDH['Deconvolution.Background'] = bg

            if beadMode and self.image.data.shape[3] > 1:
                #special case for deconvolving multi-channel PSFs
                data = np.concatenate([self.image.data[:,:,:, i].astype('f') - dlg.GetOffset() for i in range(self.image.data.shape[3])], 0)
            else:
                data = self.image.data[:,:,:, dlg.GetChannel()].astype('f') - dlg.GetOffset()

            if dlg.GetPadding():
                padsize = numpy.array(dlg.GetPadSize())
                decMDH['Deconvolution.Padding'] = padsize
                dp = numpy.ones(numpy.array(data.shape) + 2*padsize, 'f')*data.mean()
                weights = numpy.zeros_like(dp)
                px, py, pz = padsize

                dp[px:-px, py:-py, pz:-pz] = data
                weights[px:-px, py:-py, pz:-pz] = 1.

                weights = weights.ravel()
            else:
                dp = data
                weights = 1


            if dlg.GetBlocking():
                decMDH['Deconvolution.Method'] = 'Blocked ICTM'
                self.checkTQ()
                from PYME.Deconv import tq_block_dec
                bs = dlg.GetBlockSize()
                self.decT = tq_block_dec.blocking_deconv(self.tq, data, psf, self.image.seriesName, blocksize={'y': bs, 'x': bs, 'z': 256})
                self.decT.go()

            else:
                decMDH['Deconvolution.Method'] = dlg.GetMethod()
                if dlg.GetMethod() == 'ICTM':
                    decMDH['Deconvolution.RegularisationParameter'] = regLambda
                    if beadMode:
                        self.dec = dec.dec_bead()
                    else:
                        self.dec = dec.dec_conv()
                else:
                    if beadMode:
                        self.dec = richardsonLucy.rlbead()
                    else:
                        self.dec = richardsonLucy.dec_conv()

                self.dec.psf_calc(psf, dp.shape)

                self.decT = decThread.decThread(self.dec, dp, regLambda, nIter, weights, bg = bg)
                self.decT.start()

                tries = 0
                while not hasattr(self.dec, 'fs'):
                    if tries > 60:
                        self.decT.kill()
                        raise RuntimeError('Initialization not complete after 60s, giving up.')
                    
                    time.sleep(1)
                    tries += 1

                if dlg.GetPadding() and dlg.GetRemovePadding():
                    fs =  self.dec.fs[px:-px, py:-py, pz:-pz]
                else:
                    fs = self.dec.fs

                if beadMode and self.image.data.shape[3] > 1:
                    #special case for bead deconvolution - unwind the stacking
                    nChans = self.image.data.shape[3]
                    xs = int(fs.shape[0]/nChans)
                    #NOTE: this relies on the slicing returning a view into the underlying data, not a copy

                    fs = [fs[(i*xs):((i+1)*xs), :, :] for i in range(nChans)]
                
                im = ImageStack(data = fs, mdh = decMDH, titleStub = 'Deconvolution Result')
                mode = 'lite'
                if beadMode and im.data.shape[1] < 100:
                    mode = 'psf'
                    im.defaultExt = '*.psf' #we want to save as PSF by default
                self.res = ViewIm3D(im, mode=mode, parent=wx.GetTopLevelParent(self.dsviewer))                    
                    
                #self.res = View3D(fs, 'Deconvolution Result', mdh=decMDH, parent=wx.GetTopLevelParent(self.dsviewer), mode=mode)

                self.dlgDeconProg = DeconvProgressPanel(self.res, nIter)

                self.pinfo1 = aui.AuiPaneInfo().Name("deconvPanel").Top().Caption('Deconvolution Progress').DestroyOnClose(True).CloseButton(False)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
                self.res._mgr.AddPane(self.dlgDeconProg, self.pinfo1)
                self.res._mgr.Update()

            self.deconTimer = mytimer()
            self.deconTimer.WantNotification.append(self.OnDeconTimer)

            self.deconTimer.Start(500)
            
    def OnDeconvMovie(self, event, beadMode=False):
        from PYME.Deconv.deconvDialogs import DeconvSettingsDialog #,DeconvProgressDialog,DeconvProgressPanel
        #import multiprocessing

        dlg = DeconvSettingsDialog(self.dsviewer, beadMode, self.image.data.shape[3])
        if dlg.ShowModal() == wx.ID_OK:
            from PYME.Deconv import dec, richardsonLucy #, decThread
            nIter = dlg.GetNumIterationss()
            regLambda = dlg.GetRegularisationLambda()

            decMDH = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
            decMDH['Deconvolution.NumIterations'] = nIter
            decMDH['Deconvolution.OriginalFile'] = self.image.filename

            vx, vy, vz = self.image.voxelsize_nm

            if beadMode:
                from PYME.Deconv import beadGen
                psf = beadGen.genBeadImage(dlg.GetBeadRadius(), (vx, vy, vz))

                decMDH['Deconvolution.BeadRadius'] = dlg.GetBeadRadius()
                
            else:
                psfFilename, psf, vs = dlg.GetPSF(vshint = vx)

                decMDH['Deconvolution.PSFFile'] = psfFilename

                if not (vs.x == vx and vs.y == vy):# and vs.z ==vz):
                    #rescale psf to match data voxel size
                    psf = ndimage.zoom(psf, [vs.x/vx, vs.y/vy, 1])

            data = np.atleast_3d(self.image.data_xyztc[:,:,:, :, dlg.GetChannel()].astype('f').squeeze()) - dlg.GetOffset()
            decMDH['Deconvolution.Offset'] = dlg.GetOffset()
            
            bg = dlg.GetBackground()
            decMDH['Deconvolution.Background'] = bg

            #crop PSF in z if bigger than stack

            #print psf.shape, data.shape
            if psf.shape[2] > data.shape[2]:
                dz = psf.shape[2] - data.shape[2]

                psf = psf[:,:,numpy.floor(dz/2):(psf.shape[2]-numpy.ceil(dz/2))]

            print((data.shape, psf.shape))


            dp = numpy.array(data)
            weights = 1

            if dlg.GetBlocking():
                decMDH['Deconvolution.Method'] = 'Blocked ICTM'
                self.checkTQ()
                from PYME.Deconv import tq_block_dec
                bs = dlg.GetBlockSize()
                self.decT = tq_block_dec.blocking_deconv(self.tq, data, psf, self.image.seriesName, blocksize={'y': bs, 'x': bs, 'z': 256})
                self.decT.go()

            else:
                decMDH['Deconvolution.Method'] = dlg.GetMethod()
                if dlg.GetMethod() == 'ICTM':
                    decMDH['Deconvolution.RegularisationParameter'] = regLambda
                    if beadMode:
                        self.dec = dec.dec_bead()
                    else:
                        self.dec = dec.dec_conv()
                else:
                    if beadMode:
                        self.dec = richardsonLucy.rlbead()
                    else:
                        self.dec = richardsonLucy.dec_conv()

                self.dec.psf_calc(psf, dp[:,:,0:1].shape)

                #print dp.__class__

                #self.decT = decThread.decThread(self.dec, dp, regLambda, nIter, weights)
                #self.decT.start()

#                p = multiprocessing.Pool()
#                slices = [(self.dec, psf, dp[:,:,i:(i+1)],regLambda, nIter, weights)  for i in range(dp.shape[2])]                
#                r = p.map(_pt, slices)
#                res = numpy.concatenate(r, 2)

                res = numpy.concatenate([self.dec.deconv(dp[:,:,i:(i+1)], regLambda, nIter, weights, bg=bg).reshape(self.dec.shape) for i in range(dp.shape[2])], 2)
                
                
                im = ImageStack(data = res, mdh = decMDH, titleStub = 'Deconvolution Result')
                mode = 'lite'
                
                self.res = ViewIm3D(im, mode=mode, parent=wx.GetTopLevelParent(self.dsviewer))                    
                    


    def OnDeconEnd(self, sucess):
        if not 'res' in dir(self):
            self.dlgDeconProg.Destroy()
        else:
            self.res._mgr.ClosePane(self.pinfo1)
            self.res._mgr.Update()
            self.res.DataChanged()
        if sucess:
            #if 'decvp' in dir(self):
            #    for pNum in range(self.dsviewer.notebook1.GetPageCount()):
            #        if self.dsviewer.notebook1.GetPage(pNum) == self.decvp:
            #            self.dsviewer.notebook1.DeletePage(pNum)
            #self.decvp = MyViewPanel(self.dsviewer.notebook1, self.decT.res)
            #self.dsviewer.notebook1.AddPage(page=self.decvp, select=True, caption='Deconvolved')
            if not 'res' in dir(self):
                View3D(self.decT.res, '< Deconvolution Result >', parent=self.dsviewer)


    def OnDeconTimer(self, caller=None):
        if 'res' in dir(self):
            self.res.update()
            #self.res.Refresh()
            #self.res.do.Optimise()
        if self.decT.isAlive():
            if not self.dlgDeconProg.Tick(self.dec):
                self.decT.kill()
                self.OnDeconEnd(False)
        else:
            self.deconTimer.Stop()
            self.OnDeconEnd(True)

    def saveDeconvolution(self, event=None):
        fdialog = wx.FileDialog(None, 'Save Positions ...',
            wildcard='TIFF Files|*.tif', defaultFile=os.path.splitext(self.seriesName)[0] + '_dec.tif', style=wx.FD_SAVE|wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath()

            from PYME.IO.FileUtils import saveTiffStack

            saveTiffStack.saveTiffMultipage(self.dec.res, outFilename)

    def update(self, dsviewer):
        if 'decvp' in dir(self):
            self.decvp.imagepanel.Refresh()
            
    def OnDeconvWiener(self, event):
        #from PYME.Deconv import weiner

        decMDH = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
        decMDH['Deconvolution.OriginalFile'] = self.image.filename
        decMDH['Deconvolution.Method'] = 'Wiener'
        
        im = numpy.zeros(self.image.data.shape[:3], 'f4')

        decView = View3D(im, 'Deconvolution Result', mdh=decMDH, parent=self.dsviewer)

        decView.wienerPanel = WienerDeconvolver(decView, self.image, decView.image)

        self.pinfo1 = aui.AuiPaneInfo().Name("wienerPanel").Left().Caption('Wiener Filter').DestroyOnClose(True).CloseButton(False)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        decView._mgr.AddPane(decView.wienerPanel, self.pinfo1)
        decView._mgr.Update()
        
        self.dsviewer.decView = decView

import threading
class WienerDeconvolver(wx.Panel):
    def __init__(self, parent, sourceImage, destImage):
        from PYME.Deconv import wiener        
        
        wx.Panel.__init__(self, parent)
        

        self.dw = wiener.dec_wiener()
        self.sourceImage = sourceImage
        self.destImage = destImage
        
        self.havePSF = False
        self.decT = None
        
        
        #GUI stuff        
        sizer2 = wx.BoxSizer(wx.VERTICAL)
        
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)

        sizer3.Add(wx.StaticText(self, -1, 'PSF:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.fpPSF = wx.FilePickerCtrl(self, -1, wildcard='*.psf|*.tif', style=wx.FLP_OPEN|wx.FLP_FILE_MUST_EXIST)
        self.fpPSF.Bind(wx.EVT_FILEPICKER_CHANGED, self.OnPSFFileChanged)

        sizer3.Add(self.fpPSF, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(self, -1, 'Offset:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tOffset = wx.TextCtrl(self, -1, '0')

        sizer3.Add(self.tOffset, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(self, -1, u'Regularisation \u03BB:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tRegLambda = wx.TextCtrl(self, -1, '1e-8')

        sizer3.Add(self.tRegLambda, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.AddSpacer(10)
        self.bCalculate = wx.Button(self, -1, 'Apply')
        self.bCalculate.Bind(wx.EVT_BUTTON, self.OnCalculate)
        self.bCalculate.Enable(False)

        sizer3.Add(self.bCalculate, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        self.SetSizerAndFit(sizer2)
        
        self.deconTimer = mytimer()
        self.deconTimer.WantNotification.append(self.OnDeconTimer)

        self.deconTimer.Start(500)
        
    def OnPSFFileChanged(self, event=None):
        from scipy import ndimage
        
        
        vx = self.sourceImage.mdh.getEntry('voxelsize.x')
        vy = self.sourceImage.mdh.getEntry('voxelsize.y')
        vz = self.sourceImage.mdh.getEntry('voxelsize.z')

        psf, vs = numpy.load(self.fpPSF.GetPath())

        self.destImage.mdh['Deconvolution.PSFFile'] = self.fpPSF.GetPath()

        if not (vs.x == vx and vs.y == vy and vs.z ==vz):
            #rescale psf to match data voxel size
            psf = ndimage.zoom(psf, [vs.x/vx, vs.y/vy, vs.z/vz])

        #crop PSF in z if bigger than stack
        if psf.shape[2] > self.sourceImage.data.shape[2]:
            dz = psf.shape[2] - self.sourceImage.data.shape[2]

            psf = psf[:,:,numpy.floor(dz/2):(psf.shape[2]-numpy.ceil(dz/2))]
            
        print((psf.shape, self.sourceImage.data.shape))
            
        self.dw.psf_calc(psf, self.sourceImage.data.shape)
        
        self.bCalculate.Enable(os.path.exists(self.fpPSF.GetPath()))
        self.havePSF = True
        
        
    def _doDeconv(self, off, lamb):
        self.destImage.data[:] = self.dw.deconv(self.sourceImage.data[:,:,:,0].squeeze().astype('f') - off, lamb, True)
    
    def OnCalculate(self, event=None):
        if self.havePSF and not self.decT:
            lamb = float(self.tRegLambda.GetValue())
            off = float(self.tOffset.GetValue())
            
            self.decT = threading.Thread(target=self._doDeconv, args=(off, lamb))
            self.decT.start()
            
            #self.GetParent().do.Optimise()
            
    def OnDeconTimer(self, caller=None):
        if self.decT and not self.decT.isAlive():
            self.decT = None
            self.GetParent().Refresh()
        
        
    





def Plug(dsviewer):
    return Deconvolver(dsviewer)
