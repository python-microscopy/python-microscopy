# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 14:48:07 2011

@author: dbad004
"""

import wx
#from PYME.Acquire.Hardware import splitter
from PYME.Analysis.DataSources import UnsplitDataSource
import numpy as np
#from PYME.DSView.arrayViewPanel import *

class UnMixSettingsPanel(wx.Panel):
    def __init__(self, parent, splitter = None, size=(-1, -1)):
        wx.Panel.__init__(self,parent, -1, size=size)

        self.splitter = splitter

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.op = OptionsPanel(self, splitter.f.vp.do, horizOrientation=True)
        vsizer.Add(self.op, 0, wx.ALL, 0)

        psizer = wx.BoxSizer(wx.HORIZONTAL)

        bsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Mix Matrix'), wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tMM00 = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.mixMatrix[0,0]), size=(40,-1))
        hsizer.Add(self.tMM00, 1, wx.ALL,2 )

        self.tMM01 = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.mixMatrix[0,1]), size=(40,-1))
        hsizer.Add(self.tMM01, 1, wx.ALL,2 )

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tMM10 = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.mixMatrix[1,0]), size=(40,-1))
        hsizer.Add(self.tMM10, 1, wx.ALL,2 )

        self.tMM11 = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.mixMatrix[1,1]), size=(40,-1))
        hsizer.Add(self.tMM11, 1, wx.ALL,2 )

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        psizer.Add(bsizer, 0, wx.ALL, 0)


        bsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Offset'), wx.HORIZONTAL)
        self.tOffset = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.offset), size=(40,-1))
        self.bGrabOffset = wx.Button(self, -1, 'C', style = wx.BU_EXACTFIT)

        bsizer.Add(self.tOffset, 1, wx.ALL, 0)
        bsizer.Add(self.bGrabOffset, 0, wx.LEFT, 5)
        psizer.Add(bsizer, 1, wx.LEFT|wx.RIGHT, 5)


        vsizer.Add(psizer, 1, wx.ALL|wx.EXPAND, 0)
        self.SetSizerAndFit(vsizer)


    def OnUpdateMix(self, event=None):
        self.splitter.mixMatrix[0,0]= float(self.tMM00.GetValue())
        self.splitter.mixMatrix[0,1]= float(self.tMM01.GetValue())
        self.splitter.mixMatrix[1,0]= float(self.tMM10.GetValue())
        self.splitter.mixMatrix[1,1]= float(self.tMM11.GetValue())
        self.splitter.offset= float(self.tOffset.GetValue())

                                       
    
class Unmixer:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.image = dsviewer.image 
        

        EXTRAS_UNMUX = wx.NewId()
        dsviewer.mProcessing.Append(EXTRAS_UNMUX, "&Unmix", "", wx.ITEM_NORMAL)
        wx.EVT_MENU(dsviewer, EXTRAS_UNMUX, self.OnUnmix)

    def OnUnmix(self, event):
        #from PYME.Analysis import deTile
        from PYME.DSView import ViewIm3D, ImageStack

        mdh = self.image.mdh
        if 'chroma.dx' in mdh.getEntryNames():
            sf = (mdh['chroma.dx'], mdh['chroma.dy'])
        else:
            sf = None

        flip = True
        if 'Splitter.Flip' in mdh.getEntryNames():
            flip = False

        if 'Camera.ADOffset' in mdh.getEntryNames():
            ado = mdh['Camera.ADOffset']
        else:
            ado = 0

        unmux = splitter.Unmixer(sf, 1e3*mdh['voxelsize.x'], flip)
        mixMatrix = np.array([[1.,0.],[0.,1.]])

        ROIX1 = mdh.getEntry('Camera.ROIPosX')
        ROIY1 = mdh.getEntry('Camera.ROIPosY')

        ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
        ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')

        um0 = UnsplitDataSource.DataSource(self.image.data, unmux,
                                           mixMatrix, ado,
                                           [ROIX1, ROIY1, ROIX2, ROIY2], 0)

        um1 = UnsplitDataSource.DataSource(self.image.data, unmux,
                                           mixMatrix, ado,
                                           [ROIX1, ROIY1, ROIX2, ROIY2], 1)
            
        im = ImageStack([um0, um1], titleStub = 'Unmixed Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.GaussianFilter'] = sigmas

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas)


def Plug(dsviewer):
    dsviewer.unmux = Unmixer(dsviewer)
                                       
    