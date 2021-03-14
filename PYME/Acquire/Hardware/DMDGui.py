#!/usr/bin/python

#!/usr/bin/env python

import os
import wx
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import logging
logger = logging.getLogger(__name__)

display_mode = [
        'DISP_MODE_IMAGE',      #/* Static Image */
        'DISP_MODE_TEST_PTN',       #/* Internal Test pattern */
        'DISP_MODE_VIDEO',      #/* HDMI Video */
        'DISP_MODE_PTN_SEQ',]        #/* Pattern Sequence */

mode_panel = {
        'Static Image':'select static image',
        'Test Pattern':'select test pattern',
        'Image Sequence':'select image sequence',
        'Video Output':'select video source'}

test_pattern = [
        'CHECKERBOARD',
        'SOLID_BLACK',
        'SOLID_WHITE',
        'SOLID_GREEN',
        'SOLID_BLUE',
        'SOLID_RED',
        'VERT_LINES',    
        'HORIZ_LINES',
        'VERT_FINE_LINES',
        'HORIZ_FINE_LINES',
        'DIAG_LINES',
        'VERT_RAMP',
        'HORIZ_RAMP',
        'ANSI_CHECKERBOARD']

class DMDModeChooserPanel(wx.Panel):
    def __init__(self, parent, scope, **kwargs):
        wx.Panel.__init__(self, parent, **kwargs)

        self.scope = scope
        self.mf = parent # self.mf is 'MainFrame'
        
        self.DMDMode = [ 'Static Image', 'Test Pattern', 'Image Sequence']

        
        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.cDMD = wx.Choice(self, -1, choices = self.DMDMode)
        self.cDMD.SetSelection(0)
        self.cDMD.Bind(wx.EVT_CHOICE, self.OnCDMD)

        
        vsizer.Add(self.cDMD, 1, flag=wx.EXPAND|wx.LEFT|wx.RIGHT, border=10)

        vsizer.Add((-1, 25))

        self.btn = wx.Button(self, label='Set mode', style=wx.BU_EXACTFIT)
        self.btn.Bind(wx.EVT_BUTTON, self.OnBSetmodeButton)
        vsizer.Add(self.btn, flag=wx.ALIGN_RIGHT|wx.RIGHT, border=10)

        self.SetSizerAndFit(vsizer)

    def OnCDMD(self, event):
        logger.debug("Set display mode to: %s"  % self.cDMD.GetStringSelection())

    def OnBSetmodeButton(self, event):
        if self.cDMD.GetStringSelection() == 'Image Sequence':
            self.scope.LC.SetDisplayMode(self.DMDMode.index(self.cDMD.GetStringSelection())+2) # from programming guide: DMD display mode 0x00-static Image; 0x01-Internal Test Pattern; 0x02-HDMI Video Input; 0x03-Reserved; 0x04-Pattern Sequence Display.
        else:
            self.scope.LC.SetDisplayMode(self.DMDMode.index(self.cDMD.GetStringSelection()))
        for k in range(len(self.mf.camPanels)):
            if self.mf.camPanels[k][1] in ['select test pattern', 'select static image', 'select image sequence']:
                self.mf.camPanels[k][0].GetParent().Hide()
        
        for k in range(len(self.mf.camPanels)):
            if self.mf.camPanels[k][1] == mode_panel[self.cDMD.GetStringSelection()]:
                self.mf.camPanels[k][0].GetParent().Show()
                self.mf.camPanels[k][0].GetParent().GetParent().Layout()


class DMDTestPattern(wx.Panel):
    def __init__(self, parent, lightcrafter, winid=-1):
        wx.Panel.__init__(self, parent, winid)

        self.lc = lightcrafter

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.cDMDtp = wx.Choice(self, -1, choices = test_pattern)
        self.cDMDtp.SetSelection(0)
        self.cDMDtp.Bind(wx.EVT_CHOICE, self.OnCDMDtp)

        vsizer.Add(self.cDMDtp, 1, flag=wx.EXPAND|wx.LEFT|wx.RIGHT, border=10)

        vsizer.Add((-1, 25))

        self.btn = wx.Button(self, label='send', style=wx.BU_EXACTFIT)
        self.btn.Bind(wx.EVT_BUTTON, self.OnBSendButton)
        vsizer.Add(self.btn, flag=wx.ALIGN_RIGHT|wx.RIGHT, border=10)

        self.SetSizerAndFit(vsizer)

    def OnCDMDtp(self, event):
        logger.debug("select pattern %s" % self.cDMDtp.GetStringSelection())

    def OnBSendButton(self, event):
        self.lc.SetTestPattern(test_pattern.index(self.cDMDtp.GetStringSelection()))


class DMDStaticImage(wx.Panel):
    def __init__(self, parent, lightcrafter, winid=-1):
        wx.Panel.__init__(self, parent, winid)

        self.lc = lightcrafter

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.btn1 = wx.Button(self, label='Browse', style=wx.BU_EXACTFIT)
        self.btn1.Bind(wx.EVT_BUTTON, self.OnBBrowseButton)

        self.btn2 = wx.Button(self, label='Preview', style=wx.BU_EXACTFIT)
        self.btn2.Bind(wx.EVT_BUTTON, self.OnBPreviewButton)

        self.btn3 = wx.Button(self, label='Send', style=wx.BU_EXACTFIT)
        self.btn3.Bind(wx.EVT_BUTTON, self.OnBSendButton)



        hsizer.Add(self.btn1, flag=wx.ALIGN_LEFT|wx.LEFT, border=10)
        hsizer.Add((10,-1))
        hsizer.Add(self.btn2, flag=wx.ALIGN_LEFT|wx.LEFT, border=10)
        hsizer.Add((10,-1))
        hsizer.Add(self.btn3, flag=wx.ALIGN_LEFT|wx.LEFT, border=10)

        self.SetSizerAndFit(hsizer)

    def OnBBrowseButton(self, event):
        dlg = wx.FileDialog(self, message="Open an Image...", defaultDir=os.getcwd(), 
                            defaultFile="", style=wx.FD_OPEN)

        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()            
            im = Image.open(filename)
            self.data = np.asarray(im)

        dlg.Destroy()

    def OnBPreviewButton(self, event):
        plt.imshow(self.data, cmap='gray')

    def OnBSendButton(self, event):
        self.lc.SetImage(((self.data > 0)*255).astype('uint8'))


class DMDImageSeq(wx.Panel):
    def __init__(self, parent, lightcrafter, winid=-1):
        wx.Panel.__init__(self, parent, winid)

        self.lc = lightcrafter

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.btn1 = wx.Button(self, label='Browse', style=wx.BU_EXACTFIT)
        self.btn1.Bind(wx.EVT_BUTTON, self.OnBBrowseButton)

        self.btn2 = wx.Button(self, label='Send', style=wx.BU_EXACTFIT)
        self.btn2.Bind(wx.EVT_BUTTON, self.OnBSendButton)

        hsizer.Add(self.btn1, flag=wx.ALIGN_LEFT|wx.LEFT, border=10)
        hsizer.Add((10,-1))
        hsizer.Add(self.btn2, flag=wx.ALIGN_LEFT|wx.LEFT, border=10)

        self.SetSizerAndFit(hsizer)

    def OnBBrowseButton(self, event):
        self.seq = []
        dlg = wx.FileDialog(self, message="Select image sequence...", defaultDir=os.getcwd(), 
                            defaultFile="", style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            filelist = dlg.GetPaths()
        
            for filename in filelist:
                im = Image.open(filename)
                data = np.asarray(im)
                self.seq.append(data)

        dlg.Destroy()

    def OnBSendButton(self, event):
        self.lc.SetPatternDefs(self.seq)
        self.lc.StartPatternSeq()