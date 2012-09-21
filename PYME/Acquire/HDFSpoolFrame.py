#!/usr/bin/python

##################
# HDFSpoolFrame.py
#
# Copyright David Baddeley, 2009
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

#Boa:Frame:FrSpool

import wx
import datetime
import HDFSpooler
import QueueSpooler
try:
    from PYME.Acquire import sampleInformation
    sampInf = True
except:
    print 'Could not connect to the sample information database'
    sampInf = False
#import win32api
from PYME.FileUtils import nameUtils
from PYME.FileUtils.freeSpace import get_free_space
from PYME.ParallelTasks.relativeFiles import getRelFilename

import PYME.Acquire.Protocols
import PYME.Acquire.protocol as prot
import preflight

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
        APSizer.Add(self.stAqProtocol, 5,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 2)

        self.bSetAP = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetAP.Bind(wx.EVT_BUTTON, self.OnBSetAqProtocolButton)

        APSizer.Add(self.bSetAP, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        vsizer.Add(APSizer, 0, wx.ALL|wx.EXPAND, 0)
        
        

        ### Series Name & start button
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        #hsizer.Add(wx.StaticText(self, -1, 'Series: '), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tcSpoolFile = wx.TextCtrl(self, -1, 'dd_mm_series_a', size=wx.Size(100, -1))
        self.tcSpoolFile.Bind(wx.EVT_TEXT, self.OnTcSpoolFileText)

        hsizer.Add(self.tcSpoolFile, 5,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        self.bStartSpool = wx.Button(self,-1,'Series',style=wx.BU_EXACTFIT)
        self.bStartSpool.Bind(wx.EVT_BUTTON, self.OnBStartSpoolButton)
        hsizer.Add(self.bStartSpool, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        self.bStartStack = wx.Button(self,-1,'Z-Series',style=wx.BU_EXACTFIT)
        self.bStartStack.Bind(wx.EVT_BUTTON, self.OnBStartStackButton)
        hsizer.Add(self.bStartStack, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        vsizer.Add(hsizer, 0, wx.LEFT|wx.RIGHT|wx.EXPAND, 0)

        self.stDiskSpace = wx.StaticText(self, -1, 'Free space:')
        vsizer.Add(self.stDiskSpace, 0, wx.ALL|wx.EXPAND, 2)
        

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

        vsizer.Add(spoolProgSizer, 0, wx.ALL|wx.EXPAND, 0)


        ###Spool directory
        sbSpoolDir = wx.StaticBox(self, -1,'Spool Directory')
        spoolDirSizer = wx.StaticBoxSizer(sbSpoolDir, wx.HORIZONTAL)

        self.stSpoolDirName = wx.StaticText(self, -1,'Save images in: Blah Blah', size=wx.Size(136, -1))
        spoolDirSizer.Add(self.stSpoolDirName, 5,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        self.bSetSpoolDir = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetSpoolDir.Bind(wx.EVT_BUTTON, self.OnBSetSpoolDirButton)
        
        spoolDirSizer.Add(self.bSetSpoolDir, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(spoolDirSizer, 0, wx.ALL|wx.EXPAND, 0)

        ### Queue & Compression
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.cbQueue = wx.CheckBox(self, -1,'Save to Queue')
        self.cbQueue.SetValue(True)

        hsizer.Add(self.cbQueue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.cbCompress = wx.CheckBox(self, -1, 'Compression')
        self.cbCompress.SetValue(True)

        hsizer.Add(self.cbCompress, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.LEFT|wx.RIGHT|wx.EXPAND, 0)

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
        self.protocolZ = prot.NullZProtocol

        
        #if we've had to quit for whatever reason start where we left off
        while os.path.exists(os.path.join(self.dirname, self.seriesName + '.h5')):
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
        
        self.stSpoolDirName.SetLabel(self.dirname)
        self.tcSpoolFile.SetValue(self.seriesName)
        self.UpdateFreeSpace()

    def _GenSeriesName(self):
        return self.seriesStub + '_' + self._NumToAlph(self.seriesCounter)

    def _NumToAlph(self, num):
        return baseconvert(num, 'ABCDEFGHIJKLMNOPQRSTUVXWYZ')

    def UpdateFreeSpace(self, event=None):
        freeGB = get_free_space(self.dirname)/1e9
        self.stDiskSpace.SetLabel('Free Space: %3.2f GB' % freeGB)
        if freeGB < 5:
            self.stDiskSpace.SetForegroundColour(wx.Colour(200, 0,0))
        else:
            self.stDiskSpace.SetForegroundColour(wx.BLACK)

        

    def OnBStartSpoolButton(self, event=None, stack=False):
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
            ans = wx.MessageBox('A series with the same name already exists', 'Error', wx.OK)
            #overwriting doesn't work ... so just bail
            #increment the series counter first, though, so hopefully we don't get the same error on the next try
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
            self.tcSpoolFile.SetValue(self.seriesName)
            #if ans == wx.NO:
            return #bail
            
        if self.cbCompress.GetValue():
            compLevel = 6
        else:
            compLevel = 0

        if stack:
            protocol = self.protocolZ
            print protocol
        else:
            protocol = self.protocol

        if not preflight.ShowPreflightResults(self, self.protocol.PreflightCheck()):
            return #bail if we failed the pre flight check, and the user didn't choose to continue

        if self.cbQueue.GetValue():
            self.queueName = getRelFilename(self.dirname + fn + '.h5')
            self.spooler = QueueSpooler.Spooler(self.scope, self.queueName, self.scope.pa, protocol, self, complevel=compLevel)
            self.bAnalyse.Enable(True)
        else:
            self.spooler = HDFSpooler.Spooler(self.scope, self.dirname + fn + '.h5', self.scope.pa, protocol, self, complevel=compLevel)

        #if stack:
        #    self.spooler.md.setEntry('ZStack', True)

        self.bStartSpool.Enable(False)
        self.bStartStack.Enable(False)
        self.bStopSpooling.Enable(True)
        self.stSpoolingTo.Enable(True)
        self.stNImages.Enable(True)
        self.stSpoolingTo.SetLabel('Spooling to ' + fn)
        self.stNImages.SetLabel('0 images spooled in 0 minutes')

        if sampInf:
            sampleInformation.getSampleData(self, self.spooler.md)

    def OnBStartStackButton(self, event=None):
        self.OnBStartSpoolButton(stack=True)
        

    def OnBStopSpoolingButton(self, event):
        self.spooler.StopSpool()
        self.bStartSpool.Enable(True)
        self.bStartStack.Enable(True)
        self.bStopSpooling.Enable(False)
        self.stSpoolingTo.Enable(False)
        self.stNImages.Enable(False)

        self.seriesCounter +=1
        self.seriesName = self._GenSeriesName() 
        self.tcSpoolFile.SetValue(self.seriesName)
        self.UpdateFreeSpace()

    def OnBAnalyse(self, event):
        if self.cbQueue.GetValue(): #queue or not
            if sys.platform == 'win32':
                subprocess.Popen('dh5view.cmd -q %s QUEUE://%s' % (self.spooler.tq.URI, self.queueName), shell=True)
            else:
                subprocess.Popen('dh5view.py -q %s QUEUE://%s' % (self.spooler.tq.URI, self.queueName), shell=True)
#        else:
#            if sys.platform == 'win32':
#                subprocess.Popen('..\\DSView\\dh5view.cmd %s' % self.spooler.filename, shell=True)
#            else:
#                subprocess.Popen('../DSView/dh5view.py %s' % self.spooler.filename, shell=True)
        
    def Tick(self):
        dtn = datetime.datetime.now()
        
        dtt = dtn - self.spooler.dtStart
        
        self.stNImages.SetLabel('%d images spooled in %d seconds' % (self.spooler.imNum, dtt.seconds))
        self.UpdateFreeSpace()

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

            self.UpdateFreeSpace()

    def OnBSetAqProtocolButton(self, event):
        protocolList = glob.glob(PYME.Acquire.Protocols.__path__[0] + '/[a-zA-Z]*.py')
        protocolList = ['<None>',] + [os.path.split(p)[-1] for p in protocolList]
        pDlg = wx.SingleChoiceDialog(self, '', 'Select Protocol', protocolList)

        if pDlg.ShowModal() == wx.ID_OK:
            pname = pDlg.GetStringSelection()
            self.stAqProtocol.SetLabel(pname)

            if pname == '<None>':
                self.protocol = prot.NullProtocol
                self.protocolZ = prot.NullZProtocol
            else:
                pmod = __import__('PYME.Acquire.Protocols.' + pname.split('.')[0],fromlist=['PYME', 'Acquire','Protocols'])
                reload(pmod) #force module to be reloaded so that changes in the protocol will be recognised

                self.protocol = pmod.PROTOCOL
                self.protocol.filename = pname
                
                self.protocolZ = pmod.PROTOCOL_STACK
                self.protocolZ.filename = pname

        pDlg.Destroy()

    def OnTcSpoolFileText(self, event):
        event.Skip()
        
