#!/usr/bin/python

##################
# HDFSpoolFrame.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:Frame:FrSpool

import wx
import datetime
import HDFSpooler
import QueueSpooler
import sampleInformation
#import win32api
from PYME.FileUtils import nameUtils
from PYME.ParallelTasks.relativeFiles import getRelFilename

import PYME.Acquire.Protocols
import PYME.Acquire.protocol as prot

import os
import sys
import glob

import subprocess

def create(parent):
    return FrSpool(parent)

[wxID_FRSPOOL, wxID_FRSPOOLBSETSPOOLDIR, wxID_FRSPOOLBSTARTSPOOL, 
 wxID_FRSPOOLBSTOPSPOOLING, wxID_FRSPOOLCBCOMPRESS, wxID_FRSPOOLCBQUEUE, 
 wxID_FRSPOOLPANEL1, wxID_FRSPOOLSTATICBOX1, wxID_FRSPOOLSTATICBOX2, 
 wxID_FRSPOOLSTATICTEXT1, wxID_FRSPOOLSTNIMAGES, wxID_FRSPOOLSTSPOOLDIRNAME, 
 wxID_FRSPOOLSTSPOOLINGTO, wxID_FRSPOOLTCSPOOLFILE, 
] = [wx.NewId() for _init_ctrls in range(14)]

def baseconvert(number,todigits):
        x = number
    
        # create the result in base 'len(todigits)'
        res=""

        if x == 0:
            res=todigits[0]
        
        while x>0:
            digit = x % len(todigits)
            res = todigits[digit] + res
            x /= len(todigits)

        return res


class FrSpool(wx.Frame):
    def __init__(self, parent, scope, defDir, defSeries='%(day)d_%(month)d_series'):
        wx.Frame.__init__(self, id=wxID_FRSPOOL, name='FrSpool', parent=parent,
              pos=wx.Point(543, 403), size=wx.Size(285, 253),
              style=wx.DEFAULT_FRAME_STYLE, title='Spooling')
        #self.SetClientSize(wx.Size(277, 226))

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.spPan = PanSpool(self, scope, defDir, defSeries='%(day)d_%(month)d_series')

        vsizer.Add(self.spPan, 0, wx.ALL, 0)
        self.SetSizer(vsizer)
        vsizer.Fit(self)

    

class PanSpool(wx.Panel):
    def _init_ctrls(self, prnt):
        wx.Panel.__init__(self, parent=prnt, style=wx.TAB_TRAVERSAL)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        ### Aquisition Protocol
        sbAP = wx.StaticBox(self, -1,'Aquisition Protocol')
        APSizer = wx.StaticBoxSizer(sbAP, wx.HORIZONTAL)

        self.stAqProtocol = wx.StaticText(self, -1,'<None>', size=wx.Size(136, -1))
        APSizer.Add(self.stAqProtocol, 5,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        self.bSetAP = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetAP.Bind(wx.EVT_BUTTON, self.OnBSetAqProtocolButton)

        APSizer.Add(self.bSetAP, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(APSizer, 0, wx.ALL|wx.EXPAND, 5)
        
        ###Spool directory
        sbSpoolDir = wx.StaticBox(self, -1,'Spool Directory')
        spoolDirSizer = wx.StaticBoxSizer(sbSpoolDir, wx.HORIZONTAL)

        self.stSpoolDirName = wx.StaticText(self, -1,'Save images in: Blah Blah', size=wx.Size(136, -1))
        spoolDirSizer.Add(self.stSpoolDirName, 5,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        self.bSetSpoolDir = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetSpoolDir.Bind(wx.EVT_BUTTON, self.OnBSetSpoolDirButton)
        
        spoolDirSizer.Add(self.bSetSpoolDir, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(spoolDirSizer, 0, wx.ALL|wx.EXPAND, 5)

        ### Series Name & start button
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(self, -1, 'Series: '), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tcSpoolFile = wx.TextCtrl(self, -1, 'dd_mm_series_a', size=wx.Size(100, -1))
        self.tcSpoolFile.Bind(wx.EVT_TEXT, self.OnTcSpoolFileText)

        hsizer.Add(self.tcSpoolFile, 5,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        self.bStartSpool = wx.Button(self,-1,'Start',style=wx.BU_EXACTFIT)
        self.bStartSpool.Bind(wx.EVT_BUTTON, self.OnBStartSpoolButton)
        hsizer.Add(self.bStartSpool, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.LEFT|wx.RIGHT|wx.EXPAND, 5)
        
        ### Queue & Compression
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.cbQueue = wx.CheckBox(self, -1,'Save to Queue')
        self.cbQueue.SetValue(False)

        hsizer.Add(self.cbQueue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.cbCompress = wx.CheckBox(self, -1, 'Enable Compression')
        self.cbCompress.SetValue(False)

        hsizer.Add(self.cbCompress, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        vsizer.Add(hsizer, 0, wx.LEFT|wx.RIGHT|wx.EXPAND, 5)

        ### Spooling Progress

        self.sbSpoolProgress = wx.StaticBox(self, -1, 'Spooling Progress')
        self.sbSpoolProgress.Enable(False)

        spoolProgSizer = wx.StaticBoxSizer(self.sbSpoolProgress, wx.VERTICAL)

        self.stSpoolingTo = wx.StaticText(self, -1, 'Spooling to .....')
        self.stSpoolingTo.Enable(False)

        spoolProgSizer.Add(self.stSpoolingTo, 0, wx.ALL, 0)

        self.stNImages = wx.StaticText(self, -1, 'NNNNN images spooled in MM minutes')
        self.stNImages.Enable(False)

        spoolProgSizer.Add(self.stNImages, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bStopSpooling = wx.Button(self, -1, 'Stop',style=wx.BU_EXACTFIT)
        self.bStopSpooling.Enable(False)
        self.bStopSpooling.Bind(wx.EVT_BUTTON, self.OnBStopSpoolingButton)

        hsizer.Add(self.bStopSpooling, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.bAnalyse = wx.Button(self, -1, 'Analyse',style=wx.BU_EXACTFIT)
        self.bAnalyse.Enable(False)
        self.bAnalyse.Bind(wx.EVT_BUTTON, self.OnBAnalyse)

        hsizer.Add(self.bAnalyse, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        spoolProgSizer.Add(hsizer, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)

        vsizer.Add(spoolProgSizer, 0, wx.ALL|wx.EXPAND, 5)

        self.SetSizer(vsizer)
        vsizer.Fit(self)

    def __init__(self, parent, scope, defDir, defSeries='%(day)d_%(month)d_series'):
        self._init_ctrls(parent)
        self.scope = scope
        
        dtn = datetime.datetime.now()
        
        #dateDict = {'username' : win32api.GetUserName(), 'day' : dtn.day, 'month' : dtn.month, 'year':dtn.year}
        
        self.dirname = defDir % nameUtils.dateDict
        self.seriesStub = defSeries % nameUtils.dateDict

        self.seriesCounter = 0
        self.seriesName = self._GenSeriesName()

        self.protocol = prot.NullProtocol
        
        #if we've had to quit for whatever reason start where we left off
        while os.path.exists(os.path.join(self.dirname, self.seriesName + '.h5')):
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
        
        self.stSpoolDirName.SetLabel(self.dirname)
        self.tcSpoolFile.SetValue(self.seriesName)

    def _GenSeriesName(self):
        return self.seriesStub + '_' + self._NumToAlph(self.seriesCounter)

    def _NumToAlph(self, num):
        return baseconvert(num, 'ABCDEFGHIJKLMNOPQRSTUVXWYZ')
        

    def OnBStartSpoolButton(self, event):
        #fn = wx.FileSelector('Save spooled data as ...', default_extension='.log',wildcard='*.log')
        #if not fn == '': #if the user cancelled 
        #    self.spooler = Spooler.Spooler(self.scope, fn, self.scope.pa, self)
        #    self.bStartSpool.Enable(False)
        #    self.bStopSpooling.Enable(True)
        #    self.stSpoolingTo.Enable(True)
        #    self.stNImages.Enable(True)
        #    self.stSpoolingTo.SetLabel('Spooling to ' + fn)
        #    self.stNImages.SetLabel('0 images spooled in 0 minutes')
        
        fn = self.tcSpoolFile.GetValue()

        if fn == '': #sanity checking
            wx.MessageBox('Please enter a series name', 'No series name given', wx.OK)
            return #bail
        
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        if not self.dirname[-1] == os.sep:
            self.dirname += os.sep

        if (fn + '.h5') in os.listdir(self.dirname): #check to see if data with the same name exists
            ans = wx.MessageBox('A series with the same name already exists ... overwrite?', 'Warning', wx.YES_NO)
            if ans == wx.NO:
                return #bail
            
        if self.cbCompress.GetValue():
            compLevel = 6
        else:
            compLevel = 0

        if self.cbQueue.GetValue():
            self.queueName = getRelFilename(self.dirname + fn + '.h5')
            self.spooler = QueueSpooler.Spooler(self.scope, self.queueName, self.scope.pa, self.protocol, self, complevel=compLevel)
            self.bAnalyse.Enable(True)
        else:
            self.spooler = HDFSpooler.Spooler(self.scope, self.dirname + fn + '.h5', self.scope.pa, self.protocol, self, complevel=compLevel)
        self.bStartSpool.Enable(False)
        self.bStopSpooling.Enable(True)
        self.stSpoolingTo.Enable(True)
        self.stNImages.Enable(True)
        self.stSpoolingTo.SetLabel('Spooling to ' + fn)
        self.stNImages.SetLabel('0 images spooled in 0 minutes')

        sampleInformation.getSampleData(self, self.spooler.md)
        

    def OnBStopSpoolingButton(self, event):
        self.spooler.StopSpool()
        self.bStartSpool.Enable(True)
        self.bStopSpooling.Enable(False)
        self.stSpoolingTo.Enable(False)
        self.stNImages.Enable(False)

        self.seriesCounter +=1
        self.seriesName = self._GenSeriesName() 
        self.tcSpoolFile.SetValue(self.seriesName)

    def OnBAnalyse(self, event):
        if sys.platform == 'win32':
            subprocess.Popen('..\\DSView\\dh5view.cmd QUEUE://%s %s' % (self.queueName, self.spooler.tq.URI), shell=True)
        else:
            subprocess.Popen('../DSView/dh5view.py QUEUE://%s %s' % (self.queueName, self.spooler.tq.URI), shell=True)
        
    def Tick(self):
        dtn = datetime.datetime.now()
        
        dtt = dtn - self.spooler.dtStart
        
        self.stNImages.SetLabel('%d images spooled in %d seconds' % (self.spooler.imNum, dtt.seconds))

    def OnBSetSpoolDirButton(self, event):
        ndir = wx.DirSelector()
        if not ndir == '':
            self.dirname = ndir + os.sep
            self.stSpoolDirName.SetLabel(self.dirname)

            #if we've had to quit for whatever reason start where we left off
            while os.path.exists(os.path.join(self.dirname, self.seriesName + '.h5' )):
                self.seriesCounter +=1
                self.seriesName = self._GenSeriesName()
                self.tcSpoolFile.SetValue(self.seriesName)

    def OnBSetAqProtocolButton(self, event):
        protocolList = glob.glob(PYME.Acquire.Protocols.__path__[0] + '/[a-zA-Z]*.py')
        protocolList = ['<None>',] + [os.path.split(p)[-1] for p in protocolList]
        pDlg = wx.SingleChoiceDialog(self, '', 'Select Protocol', protocolList)

        if pDlg.ShowModal() == wx.ID_OK:
            pname = pDlg.GetStringSelection()
            self.stAqProtocol.SetLabel(pname)

            if pname == '<None>':
                self.protocol = prot.NullProtocol
            else:
                pmod = __import__('PYME.Acquire.Protocols.' + pname.split('.')[0],fromlist=['PYME', 'Acquire','Protocols'])
                reload(pmod) #force module to be reloaded so that changes in the protocol will be recognised

                self.protocol = pmod.PROTOCOL
                self.protocol.filename = pname

        pDlg.Destroy()

    def OnTcSpoolFileText(self, event):
        event.Skip()
        
