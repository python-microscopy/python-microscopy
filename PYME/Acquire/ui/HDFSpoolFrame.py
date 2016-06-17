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
"""The GUI controls for streaming acquisiton.

"""

import wx
import datetime

from PYME.IO.FileUtils.freeSpace import get_free_space

import PYME.Acquire.Protocols
#from PYME.Acquire.SpoolController import SpoolController

import os
import glob



[wxID_FRSPOOL, wxID_FRSPOOLBSETSPOOLDIR, wxID_FRSPOOLBSTARTSPOOL, 
 wxID_FRSPOOLBSTOPSPOOLING, wxID_FRSPOOLCBCOMPRESS, wxID_FRSPOOLCBQUEUE, 
 wxID_FRSPOOLPANEL1, wxID_FRSPOOLSTATICBOX1, wxID_FRSPOOLSTATICBOX2, 
 wxID_FRSPOOLSTATICTEXT1, wxID_FRSPOOLSTNIMAGES, wxID_FRSPOOLSTSPOOLDIRNAME, 
 wxID_FRSPOOLSTSPOOLINGTO, wxID_FRSPOOLTCSPOOLFILE, 
] = [wx.NewId() for _init_ctrls in range(14)]
    
    

class PanSpool(wx.Panel):
    """A Panel containing the GUI controls for spooling"""
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

        self.spoolProgPan = wx.Panel(self, -1)
        vsizer_sp = wx.BoxSizer(wx.VERTICAL)

        self.sbSpoolProgress = wx.StaticBox(self.spoolProgPan, -1, 'Spooling Progress')
        self.sbSpoolProgress.Enable(False)

        spoolProgSizer = wx.StaticBoxSizer(self.sbSpoolProgress, wx.VERTICAL)

        self.stSpoolingTo = wx.StaticText(self.spoolProgPan, -1, 'Spooling to .....')
        self.stSpoolingTo.Enable(False)

        spoolProgSizer.Add(self.stSpoolingTo, 0, wx.ALL, 0)

        self.stNImages = wx.StaticText(self.spoolProgPan, -1, 'NNNNN images spooled in MM minutes')
        self.stNImages.Enable(False)

        spoolProgSizer.Add(self.stNImages, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bStopSpooling = wx.Button(self.spoolProgPan, -1, 'Stop',style=wx.BU_EXACTFIT)
        self.bStopSpooling.Enable(False)
        self.bStopSpooling.Bind(wx.EVT_BUTTON, self.OnBStopSpoolingButton)

        hsizer.Add(self.bStopSpooling, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.bAnalyse = wx.Button(self.spoolProgPan, -1, 'Analyse',style=wx.BU_EXACTFIT)
        self.bAnalyse.Enable(False)
        self.bAnalyse.Bind(wx.EVT_BUTTON, self.OnBAnalyse)

        hsizer.Add(self.bAnalyse, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        spoolProgSizer.Add(hsizer, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 0)

        vsizer_sp.Add(spoolProgSizer, 0, wx.ALL|wx.EXPAND, 0)
        
        self.spoolProgPan.SetSizerAndFit(vsizer_sp)
        
        vsizer.Add(self.spoolProgPan, 0, wx.ALL|wx.EXPAND, 0)


        ###Spool directory
        sbSpoolDir = wx.StaticBox(self, -1,'Spool Directory')
        spoolDirSizer = wx.StaticBoxSizer(sbSpoolDir, wx.HORIZONTAL)

        self.stSpoolDirName = wx.StaticText(self, -1,'Save images in: Blah Blah', size=wx.Size(136, -1))
        spoolDirSizer.Add(self.stSpoolDirName, 5,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        self.bSetSpoolDir = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetSpoolDir.Bind(wx.EVT_BUTTON, self.OnBSetSpoolDirButton)
        
        spoolDirSizer.Add(self.bSetSpoolDir, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(spoolDirSizer, 0, wx.ALL|wx.EXPAND, 0)
        
        #queues etcc        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.rbQueue = wx.RadioBox(self, -1,'Spool to:', choices=['File', 'Queue', 'HTTP'])
        self.rbQueue.SetSelection(1)
        
        self.rbQueue.Bind(wx.EVT_RADIOBOX, self.OnSpoolMethodChanged)

        hsizer.Add(self.rbQueue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)


        vsizer.Add(hsizer, 0, wx.LEFT|wx.RIGHT|wx.EXPAND, 0)

        ### Queue & Compression
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        #self.cbQueue = wx.CheckBox(self, -1,'Save to Queue')
        #self.cbQueue.SetValue(True)

        #hsizer.Add(self.cbQueue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.cbCompress = wx.CheckBox(self, -1, 'Compression')
        self.cbCompress.SetValue(True)

        hsizer.Add(self.cbCompress, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.LEFT|wx.RIGHT|wx.EXPAND, 0)
        
        

        self.SetSizer(vsizer)
        vsizer.Fit(self)

    def __init__(self, parent, scope, **kwargs):
        """Initialise the spooling panel.
        
        Parameters
        ----------
        parent : wx.Window derived class
            The parent window
        scope : microscope instance
            The currently active microscope class (see microscope.py)
        defDir : string pattern
            The default directory to save data to. Any keys of the form `%(<key>)` 
            will be substituted using the values defined in `PYME.fileUtils.nameUtils.dateDict` 
        defSeries : string pattern
            This specifies a pattern for file naming. Keys will be substituted as for `defDir`
            
        """
        self._init_ctrls(parent)
        self.scope = scope
        
        #self.spoolController = SpoolController(scope, defDir, **kwargs)
        self.spoolController = scope.spoolController
        self.spoolController.onSpoolProgress.connect(self.Tick)
        self.spoolController.onSpoolStart.connect(self.OnSpoolingStarted)
        self.spoolController.onSpoolStop.connect(self.OnSpoolingStopped)

        self.stSpoolDirName.SetLabel(self.spoolController.dirname)
        self.tcSpoolFile.SetValue(self.spoolController.seriesName)
        self.UpdateFreeSpace()
        



    def UpdateFreeSpace(self, event=None):
        """Updates the free space display.
        
        Designed to be used as a callback with one of the system timers, but 
        can be called separately
        """
        freeGB = get_free_space(self.spoolController.dirname)/1e9
        self.stDiskSpace.SetLabel('Free Space: %3.2f GB' % freeGB)
        if freeGB < 5:
            self.stDiskSpace.SetForegroundColour(wx.Colour(200, 0,0))
        else:
            self.stDiskSpace.SetForegroundColour(wx.BLACK)
       

    def OnBStartSpoolButton(self, event=None, stack=False):
        """GUI callback to start spooling.
        
        NB: this is also called programatically by the start stack button."""
        
        
        fn = self.tcSpoolFile.GetValue()

        if fn == '': #sanity checking
            wx.MessageBox('Please enter a series name', 'No series name given', wx.OK)
            return #bail
            
        if self.cbCompress.GetValue():
            compLevel = 2
        else:
            compLevel = 0

        try:
            self.spoolController.StartSpooling(fn, stack=stack, compLevel = compLevel)
        except IOError:
            ans = wx.MessageBox('A series with the same name already exists', 'Error', wx.OK)
            self.tcSpoolFile.SetValue(self.spoolController.seriesName)
            
            
            
    def OnSpoolingStarted(self, **kwargs):
        if self.spoolController.spoolType in ['Queue', 'HTTP']:
            self.bAnalyse.Enable()

        self.bStartSpool.Enable(False)
        self.bStartStack.Enable(False)
        self.bStopSpooling.Enable(True)
        self.stSpoolingTo.Enable(True)
        self.stNImages.Enable(True)
        self.stSpoolingTo.SetLabel('Spooling to ' + self.spoolController.seriesName)
        self.stNImages.SetLabel('0 images spooled in 0 minutes')
        
        

    def OnBStartStackButton(self, event=None):
        """GUI callback to start spooling with z-stepping."""
        self.OnBStartSpoolButton(stack=True)
        

    def OnBStopSpoolingButton(self, event):
        """GUI callback to stop spooling."""
        self.spoolController.StopSpooling()
        #self.OnSpoolingStopped()
        
    def OnSpoolingStopped(self, **kwargs):
        self.bStartSpool.Enable(True)
        self.bStartStack.Enable(True)
        self.bStopSpooling.Enable(False)
        self.stSpoolingTo.Enable(False)
        self.stNImages.Enable(False)

        self.tcSpoolFile.SetValue(self.spoolController.seriesName)
        self.UpdateFreeSpace()

    def OnBAnalyse(self, event):
        self.spoolController.LaunchAnalysis()
        
    def Tick(self, **kwargs):
        """Called with each new frame. Updates the number of frames spooled
        and disk space remaining"""
        dtn = datetime.datetime.now()
        
        dtt = dtn - self.spoolController.spooler.dtStart
        
        self.stNImages.SetLabel('%d images spooled in %d seconds' % (self.spoolController.spooler.imNum, dtt.seconds))
        self.UpdateFreeSpace()

    def OnBSetSpoolDirButton(self, event):
        """Set the directory we're spooling into (GUI callback)."""
        ndir = wx.DirSelector()
        if not ndir == '':
            self.spoolController.SetSpoolDir(ndir)
            self.stSpoolDirName.SetLabel(self.spoolController.dirname)
            self.tcSpoolFile.SetValue(self.spoolController.seriesName)

            self.UpdateFreeSpace()

    def OnBSetAqProtocolButton(self, event):
        """Set the current protocol (GUI callback).
        
        See also: PYME.Acquire.Protocols."""
        protocolList = glob.glob(PYME.Acquire.Protocols.__path__[0] + '/[a-zA-Z]*.py')
        protocolList = ['<None>',] + [os.path.split(p)[-1] for p in protocolList]
        pDlg = wx.SingleChoiceDialog(self, '', 'Select Protocol', protocolList)

        if pDlg.ShowModal() == wx.ID_OK:
            pname = pDlg.GetStringSelection()
            self.stAqProtocol.SetLabel(pname)
            
            self.spoolController.SetProtocol(pname)

        pDlg.Destroy()

    def OnTcSpoolFileText(self, event):
        fn = self.tcSpoolFile.GetValue()
        if not fn == '':
            self.spoolController.seriesName = fn
        event.Skip()
        
    def OnSpoolMethodChanged(self, event):
        self.spoolController.SetSpoolMethod(self.rbQueue.GetStringSelection())
        self.stSpoolDirName.SetLabel(self.spoolController.dirname)
        self.tcSpoolFile.SetValue(self.spoolController.seriesName)

        self.UpdateFreeSpace()
        
