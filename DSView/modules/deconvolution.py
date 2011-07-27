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
from PYME.Acquire import MetaDataHandler

class deconvolver:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.image = dsviewer.image
        self.tq = None

        DECONV_ICTM = wx.NewId()
        DECONV_BEAD = wx.NewId()
        dsviewer.mProcessing.Append(DECONV_ICTM, "Deconvolution", "", wx.ITEM_NORMAL)
        dsviewer.mProcessing.Append(DECONV_BEAD, "Deconvolve bead shape", "", wx.ITEM_NORMAL)
        #mDeconvolution.AppendSeparator()
        #dsviewer.save_menu.Append(DECONV_SAVE, "Deconvolution", "", wx.ITEM_NORMAL)
        #self.menubar.Append(mDeconvolution, "Deconvolution")

        wx.EVT_MENU(dsviewer, DECONV_ICTM, self.OnDeconvICTM)
        wx.EVT_MENU(dsviewer, DECONV_BEAD, self.OnDeconvBead)
        #wx.EVT_MENU(dsviewer, DECONV_SAVE, self.saveDeconvolution)

        dsviewer.updateHooks.append(self.update)
        
    def checkTQ(self):
        import Pyro.core
        if self.tq == None:
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
        from PYME.Deconv.deconvDialogs import DeconvSettingsDialog,DeconvProgressDialog,DeconvProgressPanel

        dlg = DeconvSettingsDialog(self.dsviewer, beadMode)
        if dlg.ShowModal() == wx.ID_OK:
            from PYME.Deconv import dec, decThread, richardsonLucy
            nIter = dlg.GetNumIterationss()
            regLambda = dlg.GetRegularisationLambda()

            decMDH = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
            decMDH['Deconvolution.NumIterations'] = nIter
            decMDH['Deconvolution.OriginalFile'] = self.image.filename

            #self.dlgDeconProg = DeconvProgressDialog(self.dsviewer, nIter)
            #self.dlgDeconProg.Show()
            vx = self.image.mdh.getEntry('voxelsize.x')
            vy = self.image.mdh.getEntry('voxelsize.y')
            vz = self.image.mdh.getEntry('voxelsize.z')

            if beadMode:
                from PYME.Deconv import beadGen
                psf = beadGen.genBeadImage(dlg.GetBeadRadius(), (1e3*vx, 1e3*vy, 1e3*vz))

                decMDH['Deconvolution.BeadRadius'] = dlg.GetBeadRadius()
                
            else:
                psf, vs = numpy.load(dlg.GetPSFFilename())

                decMDH['Deconvolution.PSFFile'] = dlg.GetPSFFilename()

                if not (vs.x == vx and vs.y == vy and vs.z ==vz):
                    #rescale psf to match data voxel size
                    psf = ndimage.zoom(psf, [vs.x/vx, vs.y/vy, vs.z/vz])

            data = self.image.data[:,:,:].astype('f') - dlg.GetOffset()

            #crop PSF in z if bigger than stack
            if psf.shape[2] > data.shape[2]:
                dz = psf.shape[2] - data.shape[2]

                psf = psf[:,:,numpy.floor(dz/2):(psf.shape[2]-numpy.ceil(dz/2))]

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

                self.dec.psf_calc(psf, data.shape)

                self.decT = decThread.decThread(self.dec, data, regLambda, nIter)
                self.decT.start()

                tries = 0
                while tries < 10 and not hasattr(self.dec, 'fs'):
                    time.sleep(1)
                    tries += 1
                    
                self.res = View3D(self.dec.fs, '< Deconvolution Result >', mdh=decMDH, parent=self.dsviewer)

                self.dlgDeconProg = DeconvProgressPanel(self.res, nIter)

                self.pinfo1 = aui.AuiPaneInfo().Name("deconvPanel").Top().Caption('Deconvolution Progress').DestroyOnClose(True).CloseButton(False)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
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
