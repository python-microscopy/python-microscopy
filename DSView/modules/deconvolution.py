#!/usr/bin/python
##################
# blobFinding.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import wx.lib.agw.aui as aui
import numpy
from PYME.Acquire.mytimer import mytimer
from scipy import ndimage
from PYME.DSView import View3D
import time
import os

class deconvolver:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.image = dsviewer.image
        self.tq = None

        DECONV_ICTM = wx.NewId()
        DECONV_SAVE = wx.NewId()
        dsviewer.mExtras.Append(DECONV_ICTM, "Deconvolution", "", wx.ITEM_NORMAL)
        #mDeconvolution.AppendSeparator()
        #dsviewer.save_menu.Append(DECONV_SAVE, "Deconvolution", "", wx.ITEM_NORMAL)
        #self.menubar.Append(mDeconvolution, "Deconvolution")

        wx.EVT_MENU(dsviewer, DECONV_ICTM, self.OnDeconvICTM)
        #wx.EVT_MENU(dsviewer, DECONV_SAVE, self.saveDeconvolution)

        dsviewer.updateHooks.append(self.update)

    def checkTQ(self):
        import Pyro.core
        if self.tq == None:
            if 'PYME_TASKQUEUENAME' in os.environ.keys():
                taskQueueName = os.environ['PYME_TASKQUEUENAME']
            else:
                taskQueueName = 'taskQueue'
            self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)

    def OnDeconvICTM(self, event):
        from PYME.Deconv.deconvDialogs import DeconvSettingsDialog,DeconvProgressDialog,DeconvProgressPanel

        dlg = DeconvSettingsDialog(self.dsviewer)
        if dlg.ShowModal() == wx.ID_OK:
            from PYME.Deconv import dec, decThread, richardsonLucy
            nIter = dlg.GetNumIterationss()
            regLambda = dlg.GetRegularisationLambda()

            #self.dlgDeconProg = DeconvProgressDialog(self.dsviewer, nIter)
            #self.dlgDeconProg.Show()

            psf, vs = numpy.load(dlg.GetPSFFilename())

            vx = self.image.mdh.getEntry('voxelsize.x')
            vy = self.image.mdh.getEntry('voxelsize.y')
            vz = self.image.mdh.getEntry('voxelsize.z')
            
            if not (vs.x == vx and vs.y == vy and vs.z ==vz):
                #rescale psf to match data voxel size
                psf = ndimage.zoom(psf, [vx/vs.x, vy/vs.y, vz/vs.z])

            data = self.image.data[:,:,:]

            if dlg.GetBlocking():
                self.checkTQ()
                from PYME.Deconv import tq_block_dec
                bs = dlg.GetBlockSize()
                self.decT = tq_block_dec.blocking_deconv(self.tq, data, psf, self.image.seriesName, blocksize={'y': bs, 'x': bs, 'z': 256})
                self.decT.go()

            else:
                if dlg.GetMethod() == 'ICTM':
                    self.dec = dec.dec_conv()
                else:
                    self.dec = richardsonLucy.dec_conv()

                self.dec.psf_calc(psf, data.shape)

                self.decT = decThread.decThread(self.dec, data, regLambda, nIter)
                self.decT.start()

                tries = 0
                while tries < 10 and not hasattr(self.dec, 'fs'):
                    time.sleep(1)
                    tries += 1
                    
                self.res = View3D(self.dec.fs, '< Deconvolution Result >', parent=self.dsviewer)

                self.dlgDeconProg = DeconvProgressPanel(self.res, nIter)

                self.pinfo1 = aui.AuiPaneInfo().Name("deconvPanel").Top().Caption('Deconvolution Progress').DestroyOnClose(True)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
                self.res._mgr.AddPane(self.dlgDeconProg, self.pinfo1)
                self.res._mgr.Update()

            self.deconTimer = mytimer()
            self.deconTimer.WantNotification.append(self.OnDeconTimer)

            self.deconTimer.Start(500)


    def OnDeconEnd(self, sucess):
        if not 'res' in dir(self):
            self.dlgDeconProg.Destroy()
        else:
            self.res._mgr.ClosePane(self.pinfo1)
            self.res._mgr.Update()
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
            self.res.vp.do.Optimise()
        if self.decT.isAlive():
            if not self.dlgDeconProg.Tick(self.dec):
                self.decT.kill()
                self.OnDeconEnd(False)
        else:
            self.deconTimer.Stop()
            self.OnDeconEnd(True)

    def saveDeconvolution(self, event=None):
        fdialog = wx.FileDialog(None, 'Save Positions ...',
            wildcard='TIFF Files|*.tif', defaultFile=os.path.splitext(self.seriesName)[0] + '_dec.tif', style=wx.SAVE|wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            outFilename = fdialog.GetPath()

            from PYME.FileUtils import saveTiffStack

            saveTiffStack.saveTiffMultipage(self.dec.res, outFilename)

    def update(self):
        if 'decvp' in dir(self):
            self.decvp.imagepanel.Refresh()

def Plug(dsviewer):
    dsviewer.deconvolver = deconvolver(dsviewer)
