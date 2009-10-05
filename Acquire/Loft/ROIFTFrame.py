#!/usr/bin/python

##################
# ROIFTFrame.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:Frame:WebcamFrame

import wx
import wx.grid
from VideoCapture import Device
import scipy
import scipy.fftpack as fftpack
import scipy.signal
import scipy.ndimage as ndimage
import example

def create(parent):
    return WebcamFrame(parent)

[wxID_WEBCAMFRAME, wxID_WEBCAMFRAMEBMCAMVIEW, wxID_WEBCAMFRAMEBPAUSE, 
 wxID_WEBCAMFRAMEBSETPERIOD, wxID_WEBCAMFRAMEBWEBCAMSETTINGSA, 
 wxID_WEBCAMFRAMEBWEBCAMSETTINGSB, wxID_WEBCAMFRAMEEDDELAY, 
 wxID_WEBCAMFRAMEEDTHRESHOLD, wxID_WEBCAMFRAMEGRIDKVECTOR, 
 wxID_WEBCAMFRAMEMBFFT, wxID_WEBCAMFRAMESTATICBOX1, 
 wxID_WEBCAMFRAMESTATICBOX2, wxID_WEBCAMFRAMESTATICTEXT1, 
 wxID_WEBCAMFRAMESTATICTEXT2, 
] = [wx.NewId() for _init_ctrls in range(14)]

[wxID_WEBCAMFRAMEREFRESHTIMER] = [wx.NewId() for _init_utils in range(1)]

class ROIFTFrame(wx.Frame):
    def _init_utils(self):
        # generated method, don't edit
        self.refreshTimer = wx.Timer(id=wxID_WEBCAMFRAMEREFRESHTIMER,
              owner=self)
        self.Bind(wx.EVT_TIMER, self.OnRefreshTimerTimer,
              id=wxID_WEBCAMFRAMEREFRESHTIMER)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_WEBCAMFRAME, name='WebcamFrame',
              parent=prnt, pos=wx.Point(270, 118), size=wx.Size(1001, 440),
              style=wx.DEFAULT_FRAME_STYLE, title='Webcam View')
        self._init_utils()
        self.SetClientSize(wx.Size(993, 413))

        self.mbFFT = wx.StaticBitmap(bitmap=wx.NullBitmap,
              id=wxID_WEBCAMFRAMEMBFFT, name='mbFFT', parent=self,
              pos=wx.Point(368, 8), size=wx.Size(352, 288), style=0)

        self.bmCamView = wx.StaticBitmap(bitmap=wx.NullBitmap,
              id=wxID_WEBCAMFRAMEBMCAMVIEW, name='bmCamView', parent=self,
              pos=wx.Point(8, 8), size=wx.Size(352, 288), style=0)

        self.edThreshold = wx.TextCtrl(id=wxID_WEBCAMFRAMEEDTHRESHOLD,
              name='edThreshold', parent=self, pos=wx.Point(325, 328),
              size=wx.Size(100, 21), style=0, value='100')

        self.staticBox1 = wx.StaticBox(id=wxID_WEBCAMFRAMESTATICBOX1,
              label='FT Peak Detection', name='staticBox1', parent=self,
              pos=wx.Point(224, 304), size=wx.Size(216, 100), style=0)

        self.staticText2 = wx.StaticText(id=wxID_WEBCAMFRAMESTATICTEXT2,
              label='Threshold = max/', name='staticText2', parent=self,
              pos=wx.Point(240, 331), size=wx.Size(85, 13), style=0)

        self.gridKVector = wx.grid.Grid(id=wxID_WEBCAMFRAMEGRIDKVECTOR,
              name='gridKVector', parent=self, pos=wx.Point(736, 8),
              size=wx.Size(240, 400), style=0)
        self.gridKVector.EnableEditing(False)
        self.gridKVector.SetDefaultRowSize(17)
        self.gridKVector.SetDefaultColSize(40)
        self.gridKVector.SetRowLabelSize(40)

        self.staticBox2 = wx.StaticBox(id=wxID_WEBCAMFRAMESTATICBOX2,
              label='Aquisition Properties', name='staticBox2', parent=self,
              pos=wx.Point(8, 304), size=wx.Size(200, 100), style=0)

        self.edDelay = wx.TextCtrl(id=wxID_WEBCAMFRAMEEDDELAY, name='edDelay',
              parent=self, pos=wx.Point(103, 326), size=wx.Size(41, 21),
              style=0, value='500')

        self.staticText1 = wx.StaticText(id=wxID_WEBCAMFRAMESTATICTEXT1,
              label='Timer Period (ms)', name='staticText1', parent=self,
              pos=wx.Point(15, 330), size=wx.Size(83, 13), style=0)

        self.bSetPeriod = wx.Button(id=wxID_WEBCAMFRAMEBSETPERIOD, label='Set',
              name='bSetPeriod', parent=self, pos=wx.Point(157, 326),
              size=wx.Size(40, 23), style=0)
        self.bSetPeriod.Bind(wx.EVT_BUTTON, self.OnBSetPeriodButton,
              id=wxID_WEBCAMFRAMEBSETPERIOD)

        self.bWebcamSettingsA = wx.Button(id=wxID_WEBCAMFRAMEBWEBCAMSETTINGSA,
              label='Webcam Settings A', name='bWebcamSettingsA', parent=self,
              pos=wx.Point(18, 351), size=wx.Size(104, 23), style=0)
        self.bWebcamSettingsA.Bind(wx.EVT_BUTTON, self.OnBWebcamSettingsButton,
              id=wxID_WEBCAMFRAMEBWEBCAMSETTINGSA)

        self.bPause = wx.Button(id=wxID_WEBCAMFRAMEBPAUSE, label='Pause',
              name='bPause', parent=self, pos=wx.Point(145, 364),
              size=wx.Size(40, 23), style=0)
        self.bPause.Bind(wx.EVT_BUTTON, self.OnBPauseButton,
              id=wxID_WEBCAMFRAMEBPAUSE)

        self.bWebcamSettingsB = wx.Button(id=wxID_WEBCAMFRAMEBWEBCAMSETTINGSB,
              label='Webcam Settings B', name='bWebcamSettingsB', parent=self,
              pos=wx.Point(18, 376), size=wx.Size(104, 23), style=0)
        self.bWebcamSettingsB.Bind(wx.EVT_BUTTON, self.OnBWebcamSettingsBButton,
              id=wxID_WEBCAMFRAMEBWEBCAMSETTINGSB)

    def __init__(self, parent,ds):
        self._init_ctrls(parent)
        #self.webcam = Device();
        self.ds
        self.gridKVector.CreateGrid(10,5)
        self.gridKVector.SetColLabelValue(0,'Mag')
        self.gridKVector.SetColLabelValue(1,'Kx')
        self.gridKVector.SetColLabelValue(2,'Ky')
        
        self.refreshTimer.Start(500)

    def OnRefreshTimerTimer(self, event):
        self.Refresh()
        #event.Skip()
        
    def Refresh(self):
        #im = self.webcam.getImage()
        im_a = example.CDataStack_AsArray(self.ds,0)
        if (im_a.size > 1): #check to see if we actually got an image
            if (not self.__dict__.has_key('fft_window')) or not ((self.fft_window.shape[0] == im_a.shape[0]) and (self.fft_window.shape[1] == im_a.shape[1])):
                wind1 = scipy.signal.hamming(im_a.shape[0])
                wind2 = scipy.signal.hamming(im_a.shape[1])
                w1r = wind1.reshape((len(wind1),1)).repeat(len(wind2),1)
                w2r = wind2.reshape((1,len(wind2))).repeat(len(wind1),0)
                self.fft_window = w1r*w2r
                
            bm_i = wx.BitmapFromBuffer(im_a.shape[1], im_a.shape[0], scipy.concatenate((im_a, im_a, im_a),2).ravel())
            self.im_b = ndimage.gaussian_filter(scipy.mean(im_a,2, 'f'),1)*self.fft_window
            self.IM_B = scipy.absolute(fftpack.fftshift(fftpack.fft2(self.im_b - self.im_b.mean())))
            IMB = scipy.log10(self.IM_B)
            IMB = (IMB * 255/IMB.max()).astype('uint8')
            #print IMB.shape
            #print IMB.dtype
            IMB = IMB.reshape(( im_a.shape[0], im_a.shape[1],1))
            bm_ft = wx.BitmapFromBuffer(im_a.shape[1], im_a.shape[0], scipy.concatenate((IMB, IMB, IMB),2).ravel())
            
            self.bmCamView.SetBitmap(bm_i)
            self.mbFFT.SetBitmap(bm_ft)
            
            self.ExtractKVectors()
            
    def ExtractKVectors(self):
        threshold = self.IM_B.max()/float(self.edThreshold.GetValue())
        (labs, n) = scipy.ndimage.label(self.IM_B > threshold)
        self.xc = self.IM_B.shape[0]/2.0
        self.yc = self.IM_B.shape[1]/2.0

        self.pos_s = scipy.asarray(ndimage.center_of_mass(self.IM_B, labs, range(1,n+1)))
        self.maxs = scipy.asarray(ndimage.mean(self.IM_B, labs, range(1,n+1)))
        
        #maxs_c = self.maxs.copy()
        #maxs_c.sort()
        self.I = self.maxs.argsort()
        
        self.gridKVector.ClearGrid()

        if (n > self.gridKVector.GetNumberRows()):
            self.gridKVector.AppendRows(n - self.gridKVector.GetNumberRows())
        elif(n < self.gridKVector.GetNumberRows()):
            self.gridKVector.DeleteRows(0,self.gridKVector.GetNumberRows()- n)
            
        for i in range(n):
            self.gridKVector.SetCellValue(i, 0, '%3.0f' % self.maxs[self.I[n-i-1]])
            self.gridKVector.SetCellValue(i, 1, '%3.3f' % ((self.pos_s[self.I[n-i-1],0]- self.xc)/self.xc))
            self.gridKVector.SetCellValue(i, 2, '%3.3f' % ((self.pos_s[self.I[n-i-1],1]- self.yc)/self.yc))

    def OnBSetPeriodButton(self, event):
        #period = 
        self.refreshTimer.Stop()
        self.refreshTimer.Start(int(self.edDelay.GetValue()))
        #event.Skip()

    def OnBWebcamSettingsButton(self, event):
        self.webcam.displayCaptureFilterProperties()
        #event.Skip()

    def OnBPauseButton(self, event):
        if(self.refreshTimer.IsRunning()):
            self.refreshTimer.Stop()
        else:
            self.refreshTimer.Start()
        #event.Skip()

    def OnBWebcamSettingsBButton(self, event):
        self.webcam.displayCapturePinProperties()
        #event.Skip()
            
        
