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
"""The GUI controls for streaming acquisition.

"""

import wx
import datetime

import PYME.Acquire.Protocols
from PYME.Acquire.SpoolController import SpoolController
#import PYME.Acquire.protocol_acquisition

import os
import glob

from PYME.IO import PZFFormat
import sys

from PYME import config

[wxID_FRSPOOL, wxID_FRSPOOLBSETSPOOLDIR, wxID_FRSPOOLBSTARTSPOOL, 
 wxID_FRSPOOLBSTOPSPOOLING, wxID_FRSPOOLCBCOMPRESS, wxID_FRSPOOLCBQUEUE, 
 wxID_FRSPOOLPANEL1, wxID_FRSPOOLSTATICBOX1, wxID_FRSPOOLSTATICBOX2, 
 wxID_FRSPOOLSTATICTEXT1, wxID_FRSPOOLSTNIMAGES, wxID_FRSPOOLSTSPOOLDIRNAME, 
 wxID_FRSPOOLSTSPOOLINGTO, wxID_FRSPOOLTCSPOOLFILE, 
] = [wx.NewIdRef() for _init_ctrls in range(14)]
    

import  PYME.ui.manualFoldPanel as afp
from PYME.ui import cascading_layout
from . import seqdialog
from . import AnalysisSettingsUI

import logging
logger = logging.getLogger(__name__)

class ProtocolAcquisitionPane(cascading_layout.CascadingLayoutPanel):
    def _protocol_pan(self):
        pan = wx.Panel(parent=self, style=wx.TAB_TRAVERSAL)
    
        vsizer = wx.BoxSizer(wx.VERTICAL)
    
        ### Aquisition Protocol
        sbAP = wx.StaticBox(pan, -1, 'Aquisition Protocol')
        APSizer = wx.StaticBoxSizer(sbAP, wx.VERTICAL)
    
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
    
        self.stAqProtocol = wx.StaticText(pan, -1, '<None>', size=wx.Size(136, -1))
        hsizer.Add(self.stAqProtocol, 5, wx.ALL | wx.EXPAND, 2)
    
        self.bSetAP = wx.Button(pan, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetAP.Bind(wx.EVT_BUTTON, self.OnBSetAqProtocolButton)
    
        hsizer.Add(self.bSetAP, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
    
        APSizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 0)
    
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.rbNoSteps = wx.RadioButton(pan, -1, 'Standard', style=wx.RB_GROUP)
        self.rbNoSteps.Bind(wx.EVT_RADIOBUTTON, self.OnToggleZStepping)
        hsizer.Add(self.rbNoSteps, 1, wx.ALL | wx.EXPAND, 2)
        self.rbZStepped = wx.RadioButton(pan, -1, 'Z stepped')
        self.rbZStepped.Bind(wx.EVT_RADIOBUTTON, self.OnToggleZStepping)
        hsizer.Add(self.rbZStepped, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        
        if not hasattr(self.scope, 'stackSettings'):
            self.rbZStepped.Disable()
    
        if self.protocol_settings.z_stepped:
            self.rbZStepped.SetValue(True)
        else:
            self.rbNoSteps.SetValue(True)
    
        APSizer.Add(hsizer, 0, wx.TOP | wx.EXPAND, 4)
             
    
        vsizer.Add(APSizer, 0, wx.TOP | wx.EXPAND, 4)
    
        pan.SetSizerAndFit(vsizer)
        return pan
    
    def _init_ctrls(self):
        vsizer = wx.BoxSizer(wx.VERTICAL)
        vsizer.Add(self._protocol_pan(), 0, wx.ALL | wx.EXPAND, 0)

        if hasattr(self.scope, 'stackSettings'):
            clp = afp.collapsingPane(self, caption='Z stepping ...')
            self._seq_panel = seqdialog.seqPanel(clp, self.scope, mode='sequence')
            clp.AddNewElement(self._seq_panel, priority=1)
            vsizer.Add(clp, 0, wx.ALL | wx.EXPAND, 0)
            #self.AddNewElement(clp, priority=1)
            self.seq_pan = clp

        #analysis settings
        clp = afp.collapsingPane(self, caption='Real time analysis ...')

        self.scope.analysisSettings = AnalysisSettingsUI.AnalysisSettings() #Move me???

        clp.AddNewElement(AnalysisSettingsUI.AnalysisSettingsPanel(clp, self.scope.analysisSettings,
                                                                   self.scope.analysisSettings.onMetadataChanged))
        clp.AddNewElement(AnalysisSettingsUI.AnalysisDetailsPanel(clp, self.scope.analysisSettings,
                                                                  self.scope.analysisSettings.onMetadataChanged))
        #self.AddNewElement(clp)
        vsizer.Add(clp, 0, wx.ALL | wx.EXPAND, 0)
        self.SetSizerAndFit(vsizer)
        #end analysis settings
        

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
        #afp.foldingPane.__init__(self, parent, caption='Protocol Acquisition', **kwargs)
        cascading_layout.CascadingLayoutPanel.__init__(self, parent, **kwargs)
        self.scope = scope
        self.spoolController = scope.spoolController
        
        self._init_ctrls()

    @property
    def protocol_settings(self) -> PYME.Acquire.SpoolController.ProtocolAcquisitionSettings:
        return self.scope.spoolController.protocol_settings

    def OnToggleZStepping(self, event):
        self.protocol_settings.z_stepped = self.rbZStepped.GetValue()
        pan = event.GetEventObject().GetParent()
        if self.rbZStepped.GetValue():
            if self.seq_pan.folded:
                self.seq_pan.OnFold()
            
        else:
            if not self.seq_pan.folded:
                self.seq_pan.OnFold()

    def OnBSetAqProtocolButton(self, event):
        """Set the current protocol (GUI callback).
        
        See also: PYME.Acquire.Protocols."""
        from PYME.Acquire import protocol
        pDlg = wx.SingleChoiceDialog(self, '', 'Select Protocol', protocol.get_protocol_list())

        ret = pDlg.ShowModal()
        #print 'Protocol choice: ', ret, wx.ID_OK
        if ret == wx.ID_OK:
            pname = pDlg.GetStringSelection()
            self.spoolController.protocol_settings.set_protocol(pname)
            # do this after setProtocol so that an error in SetProtocol avoids setting the new name
            self.stAqProtocol.SetLabel(pname)
            self._seq_panel.UpdateDisp()  # update display of e.g. z_dwell

        pDlg.Destroy()




class SpoolingPane(afp.foldingPane):
    """A Panel containing the GUI controls for spooling"""
    
    def _spool_to_pan(self):
        pan = cascading_layout.CascadingLayoutPanel(parent=self, style=wx.TAB_TRAVERSAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
    
        ###Spool directory
        sbSpoolDir = wx.StaticBox(pan, -1, 'Spool to:')
        spoolDirSizer = wx.StaticBoxSizer(sbSpoolDir, wx.VERTICAL)
    
        #queues etcc
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self._spool_method_rbs = []
        for i, sm in enumerate(self.spoolController.available_spool_methods):
            rb = wx.RadioButton(pan, -1, sm, style=(wx.RB_GROUP if i == 0 else 0))
            rb.Bind(wx.EVT_RADIOBUTTON, self.OnSpoolMethodChanged)
            hsizer.Add(rb, 1, wx.ALL | wx.EXPAND, 2)
            self._spool_method_rbs.append(rb)

        self._spool_method_rbs[self.spoolController.available_spool_methods.index(self.spoolController.spoolType)].SetValue(True)
    
        spoolDirSizer.Add(hsizer, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 0)
    
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stSpoolDirName = wx.StaticText(pan, -1, 'Save images in: Blah Blah', size=wx.Size(136, -1))
        hsizer.Add(self.stSpoolDirName, 5, wx.ALL | wx.EXPAND, 5)
    
        self.bSetSpoolDir = wx.Button(pan, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetSpoolDir.Bind(wx.EVT_BUTTON, self.OnBSetSpoolDirButton)
    
        hsizer.Add(self.bSetSpoolDir, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
    
        spoolDirSizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 0)

        ### Series Name & start button
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
    
        #hsizer.Add(wx.StaticText(self, -1, 'Series: '), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
    
        self.tcSpoolFile = wx.TextCtrl(pan, -1, 'dd_mm_series_a', size=wx.Size(100, -1))
        self.tcSpoolFile.Bind(wx.EVT_TEXT, self.OnTcSpoolFileText)
    
        hsizer.Add(self.tcSpoolFile, 5, wx.ALL | wx.EXPAND, 5)
    
       
    
        # self.bStartStack = wx.Button(pan,-1,'Z-Series',style=wx.BU_EXACTFIT)
        # self.bStartStack.Bind(wx.EVT_BUTTON, self.OnBStartStackButton)
        # hsizer.Add(self.bStartStack, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
    
        spoolDirSizer.Add(hsizer, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 0)
    
        self.stDiskSpace = wx.StaticText(pan, -1, 'Free space:')
        spoolDirSizer.Add(self.stDiskSpace, 0, wx.ALL | wx.EXPAND, 2)

        ### Compression etc
        clp = afp.collapsingPane(pan, caption='Compression and quantization ...', padding=2)
        clp.AddNewElement(self._comp_pan(clp))
        spoolDirSizer.Add(clp, 0, wx.ALL | wx.EXPAND, 2)

        self._comp_p = clp
    
        vsizer.Add(spoolDirSizer, 0, wx.ALL | wx.EXPAND, 0)
    
        pan.SetSizerAndFit(vsizer)
        return pan
    
    def _comp_pan(self, clp):
        pan = cascading_layout.CascadingLayoutPanel(clp, -1)
    
        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
    
        self.cbCompress = wx.CheckBox(pan, -1, 'Compression')
        self.cbCompress.SetValue(self.spoolController.hdf_compression_level > 0)
    
        hsizer.Add(self.cbCompress, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
    
        self.cbQuantize = wx.CheckBox(pan, -1, 'Quantization')
        self.cbQuantize.SetValue(config.get('spooler-quantize_by_default', False))
    
        hsizer.Add(self.cbQuantize, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
    
        vsizer.Add(hsizer, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 0)
    
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Quantization offset:'), 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
    
        self.tQuantizeOffset = wx.TextCtrl(pan, -1, 'auto')
        hsizer.Add(self.tQuantizeOffset, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
    
        vsizer.Add(hsizer, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 0)
    
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Quantization scale:'), 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
    
        self.tQuantizeScale = wx.TextCtrl(pan, -1, '0.5')
        self.tQuantizeScale.SetToolTip(wx.ToolTip(
            'Quantization scale in units of sigma\n. The default of 0.5 will give a quantization interval that is half the std dev. of the expected Poisson noise in a pixel.'))
        hsizer.Add(self.tQuantizeScale, 1, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
    
        vsizer.Add(hsizer, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 0)
    
        # spool to h5 on cluster
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
    
        self.cbClusterh5 = wx.CheckBox(pan, -1, 'Spool to h5 on cluster (cluster of 1)')
        self.cbClusterh5.SetValue(self.spoolController.cluster_h5)
        self.cbClusterh5.Bind(wx.EVT_CHECKBOX, lambda e: setattr(self.spoolController,'cluster_h5', self.cbClusterh5.GetValue()))
        #self.cbClusterh5.SetValue(self._N_data_servers == 1) #set to true if we have a single node cluster
    
        hsizer.Add(self.cbClusterh5, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 0)
        pan.SetSizerAndFit(vsizer)
        
        #setup callbacks to update compression settings on UI change
        self.cbCompress.Bind(wx.EVT_CHECKBOX, lambda e: self.update_spooler_compression_settings())
        self.cbQuantize.Bind(wx.EVT_CHECKBOX, lambda e: self.update_spooler_compression_settings())
        self.tQuantizeOffset.Bind(wx.EVT_KILL_FOCUS, lambda e: self.update_spooler_compression_settings())
        self.tQuantizeScale.Bind(wx.EVT_KILL_FOCUS, lambda e: self.update_spooler_compression_settings())
        
        return pan
    
        
    def _spool_pan(self):
        pan = wx.Panel(parent=self, style=wx.TAB_TRAVERSAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
    
        ### Spooling Progress
    
        self.spoolProgPan = wx.Panel(pan, -1)
        vsizer_sp = wx.BoxSizer(wx.VERTICAL)
    
        self.sbSpoolProgress = wx.StaticBox(self.spoolProgPan, -1, 'Spooling Progress')
        self.sbSpoolProgress.Enable(False)
    
        spoolProgSizer = wx.StaticBoxSizer(self.sbSpoolProgress, wx.VERTICAL)
    
        self.stSpoolingTo = wx.StaticText(self.spoolProgPan, -1, 'Spooling to .....')
        spoolProgSizer.Add(self.stSpoolingTo, 0, wx.ALL, 0)
    
        self.stNImages = wx.StaticText(self.spoolProgPan, -1, 'NNNNN images spooled in MM minutes')
        self.stSpoolingTo.SetForegroundColour(wx.TheColourDatabase.Find('GREY'))
        self.stNImages.SetForegroundColour(wx.TheColourDatabase.Find('GREY'))
    
        spoolProgSizer.Add(self.stNImages, 0, wx.ALL, 0)
    
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bStartSpool = wx.Button(self.spoolProgPan, -1, 'Start', style=wx.BU_EXACTFIT)
        self.bStartSpool.Bind(wx.EVT_BUTTON, self.OnBStartSpoolButton)
        #self.bStartSpool.SetDefault()
        hsizer.Add(self.bStartSpool, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
    
        self.bStopSpooling = wx.Button(self.spoolProgPan, -1, 'Stop', style=wx.BU_EXACTFIT)
        self.bStopSpooling.Enable(False)
        self.bStopSpooling.Bind(wx.EVT_BUTTON, self.OnBStopSpoolingButton)
    
        hsizer.Add(self.bStopSpooling, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
    
        self.bAnalyse = wx.Button(self.spoolProgPan, -1, 'Analyse', style=wx.BU_EXACTFIT)
        self.bAnalyse.Enable(False)
        self.bAnalyse.Bind(wx.EVT_BUTTON, self.OnBAnalyse)
    
        hsizer.Add(self.bAnalyse, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
    
        spoolProgSizer.Add(hsizer, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 0)
    
        vsizer_sp.Add(spoolProgSizer, 0, wx.ALL | wx.EXPAND, 0)
    
        self.spoolProgPan.SetSizerAndFit(vsizer_sp)
    
        vsizer.Add(self.spoolProgPan, 0, wx.ALL | wx.EXPAND, 0)
    
        pan.SetSizerAndFit(vsizer)
        return pan
        
    def _aq_settings_pan(self):
        pan = cascading_layout.CascadingLayoutPanel(parent=self, style=wx.TAB_TRAVERSAL)
        v1 = wx.BoxSizer(wx.VERTICAL)
        vsizer = wx.StaticBoxSizer(wx.StaticBox(pan, label='Acquisition Type'), wx.VERTICAL)
    
        self._aq_type_btns = []

        _settings_box_name = ''

        for i, aq_type in enumerate(self.spoolController.acquisition_types):
            pane, name = self.acquisition_uis[aq_type]
            btn = wx.RadioButton(pan, -1, name, style=(wx.RB_GROUP if i == 0 else 0))
            btn.aq_type = aq_type
            btn.SetValue(aq_type == self.spoolController.acquisition_type)
            if aq_type == self.spoolController.acquisition_type:
                _settings_box_name = f'{name} Settings'
            btn.Bind(wx.EVT_RADIOBUTTON, self.OnAqTypeChanged)
            vsizer.Add(btn, 0, wx.ALL | wx.EXPAND, 2)
            self._aq_type_btns.append(btn)

    
        v1.Add(vsizer, 0, wx.ALL | wx.EXPAND, 0)
        self.sbAqSettings = wx.StaticBox(pan, -1, _settings_box_name)
        vsizer = wx.StaticBoxSizer(self.sbAqSettings, wx.VERTICAL)

        for i, aq_type in enumerate(self.spoolController.acquisition_types):
            pane, name = self.acquisition_uis[aq_type]

            pane.Reparent(pan)
            vsizer.Add(pane, 1, wx.ALL | wx.EXPAND, 2)
            pane.Show(aq_type == self.spoolController.acquisition_type)
        
        v1.Add(vsizer, 0, wx.ALL | wx.EXPAND, 0)
        pan.SetSizerAndFit(v1)

        self.aq_settings_pan = pan
        return pan
    
    def OnAqTypeChanged(self, event):
        #btn = event.GetEventObject()
        #if btn.GetValue():
        self.spoolController.acquisition_type = event.GetEventObject().aq_type
        self.on_aq_type_changed()

    def on_aq_type_changed(self):
        for btn in self._aq_type_btns:
            btn.SetValue(btn.aq_type == self.spoolController.acquisition_type)

        for aq_type in self.spoolController.acquisition_types:
            self.acquisition_uis[aq_type][0].Show(aq_type == self.spoolController.acquisition_type)
            if (aq_type == self.spoolController.acquisition_type):
                self.sbAqSettings.SetLabel(f'{self.acquisition_uis[aq_type][1]} Settings')

        self.aq_settings_pan.cascading_layout()
        #self.cascading_layout()
    
    def _init_ctrls(self):
        
        self.AddNewElement(self._aq_settings_pan(), priority=1, foldable=False)
        self._pan_spool_to = self._spool_to_pan()
        self.AddNewElement(self._pan_spool_to)

        self.AddNewElement(self._spool_pan())

        
        

    def __init__(self, parent, scope, acquisition_uis = {}, **kwargs):
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
        afp.foldingPane.__init__(self, parent, caption='Spooling', **kwargs)
        self.scope = scope
        self.spoolController = scope.spoolController
        self.acquisition_uis = acquisition_uis

        #check to see if we have a cluster
        #self._N_data_servers = len(hybrid_ns.getNS('_pyme-http').get_advertised_services())
        
        self._init_ctrls()
        
        #self.spoolController = SpoolController(scope, defDir, **kwargs)
        
        self.spoolController.onSpoolProgress.connect(self._tick)
        self.spoolController.onSpoolStart.connect(self.OnSpoolingStarted)
        self.spoolController.on_stop.connect(self.OnSpoolingStopped)

        self.stSpoolDirName.SetLabel(self.spoolController.display_dirname)
        self.tcSpoolFile.SetValue(self.spoolController.seriesName)
        self.UpdateFreeSpace()

        #update the spool method (specifically so that the default in the GUI and spool controller match)
        #self.OnSpoolMethodChanged(None)
        
        #make default compression settings in spooler match the display.
        self.update_spooler_compression_settings(False)
        

    def update_spooler_compression_settings(self, ui_on_error=True):
        try:
            self.spoolController.pzf_compression_settings = self.get_compression_settings(ui_on_error)
            self.spoolController.hdf_compression_level = 2 if self.cbCompress.GetValue() else 0
        except:
            logger.warn('Compression settings invalid, disabling quantization')
            if ui_on_error:
                ans = wx.MessageBox(
                    "Compression settings invalid, disabling quantization",
                    'Error', wx.OK)
            self.cbQuantize.SetValue(False)
            self.spoolController.pzf_compression_settings = self.get_compression_settings()
    
    def _get_spool_method(self):
        for i, rb in enumerate(self._spool_method_rbs):
            if rb.GetValue():
                return self.spoolController.available_spool_methods[i]
        
        return None

    def UpdateFreeSpace(self, event=None):
        """Updates the free space display.
        
        Designed to be used as a callback with one of the system timers, but 
        can be called separately
        """
        freeGB = self.spoolController.get_free_space()
        self.stDiskSpace.SetLabel('Free Space: %3.2f GB' % freeGB)
        if freeGB < 5:
            self.stDiskSpace.SetForegroundColour(wx.Colour(200, 0,0))
        else:
            self.stDiskSpace.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
       
    
                
    def get_compression_settings(self, ui_message_on_error=True):
        if not self.cbQuantize.GetValue():
            compSettings = {
                'compression': PZFFormat.DATA_COMP_HUFFCODE if self.cbCompress.GetValue() else PZFFormat.DATA_COMP_RAW,
                'quantization': PZFFormat.DATA_QUANT_NONE,
                'quantizationOffset': 0.0,
                'quantizationScale': 1.0
            }
    
            return compSettings
        
        else:
            #try and set our quantization offset automatically as the AD offset of the camera
            q_offset = self.tQuantizeOffset.GetValue()
            if q_offset == 'auto':
                #FIXME - add getter to camera???
                try:
                    q_offset = self.scope.cam.noise_properties['ADOffset']
                except AttributeError:
                    if ui_message_on_error:
                        ans = wx.MessageBox(
                                "Camera doesn't define noise properties, manually set the desired quantization offset",
                                'Error', wx.OK)
                    raise
            else:
                q_offset = float(q_offset)
        
            #quantization scale in GUI is in units of sigma, convert to ADU
            try:
                q_scale = float(self.tQuantizeScale.GetValue()) / self.scope.cam.noise_properties['ElectronsPerCount']
            except (AttributeError, NotImplementedError):
                logger.warning("Camera doesn't provide electrons per count, using qscale in units of ADUs instead")
                q_scale = float(self.tQuantizeScale.GetValue())
        
            compSettings = {
                'compression': PZFFormat.DATA_COMP_HUFFCODE if self.cbCompress.GetValue() else PZFFormat.DATA_COMP_RAW,
                'quantization': PZFFormat.DATA_QUANT_SQRT,
                'quantizationOffset': q_offset,
                'quantizationScale': q_scale
            }
            
            return compSettings
            

    def OnBStartSpoolButton(self, event=None, stack=False):
        """GUI callback to start spooling.
        
        NB: this is also called programatically by the start stack button."""
        
        #if self.rbZStepped.GetValue():
        #    stack = True
        
        fn = self.tcSpoolFile.GetValue()

        if fn == '': #sanity checking
            wx.MessageBox('Please enter a series name', 'No series name given', wx.OK)
            return #bail
            
        # if self.cbCompress.GetValue():
        #     compLevel = 2
        # else:
        #     compLevel = 0
        

        try:
            self.spoolController.start_spooling(fn)
        except IOError as e:
            logger.exception('IO error whilst spooling')
            ans = wx.MessageBox(str(e.strerror), 'Error', wx.OK)
            self.tcSpoolFile.SetValue(self.spoolController.seriesName)
            
    def update_ui(self):
        self.cbCompress.SetValue(self.spoolController.hdf_compression_level > 0)
            
            
            
    def OnSpoolingStarted(self, **kwargs):
        if self.spoolController.spoolType in ['Queue', 'Cluster']:
            self.bAnalyse.Enable()

        self.bStartSpool.Enable(False)
        #self.bStartStack.Enable(False)
        self.bStopSpooling.Enable(True)
        #self.stSpoolingTo.Enable(True)
        #self.stNImages.Enable(True)
        self.stSpoolingTo.SetForegroundColour(None)
        self.stNImages.SetForegroundColour(None)
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
        #self.bStartStack.Enable(True)
        self.bStopSpooling.Enable(False)
        #self.stSpoolingTo.Enable(False)
        #self.stNImages.Enable(False)
        self.stSpoolingTo.SetForegroundColour(wx.TheColourDatabase.Find('GREY'))
        self.stNImages.SetForegroundColour(wx.TheColourDatabase.Find('GREY'))

        self.stSpoolDirName.SetLabel(self.spoolController.display_dirname)
        self.tcSpoolFile.SetValue(self.spoolController.seriesName)
        self.UpdateFreeSpace()
        
        
        wx.CallAfter(self.Refresh)
        #self.Refresh()
        #self.Update()
        #self.cascading_layout()

    def OnBAnalyse(self, event):
        self.spoolController.LaunchAnalysis()
        
    
    def _tick(self, **kwargs):
        wx.CallAfter(self.Tick)

    def Tick(self, **kwargs):
        """Called with each new frame. Updates the number of frames spooled
        and disk space remaining"""
        dtn = datetime.datetime.now()
        
        dtt = dtn - self.spoolController.spooler.dtStart
        
        self.stNImages.SetLabel('%d images spooled in %d seconds' % (self.spoolController.spooler.frame_num, dtt.seconds))
        self.UpdateFreeSpace()

    def OnBSetSpoolDirButton(self, event):
        """Set the directory we're spooling into (GUI callback)."""
        ndir = wx.DirSelector()
        if not ndir == '':
            logger.debug('series name %s' % self.spoolController.seriesName)
            self.spoolController.SetSpoolDir(ndir)
            self.stSpoolDirName.SetLabel(self.spoolController.display_dirname)
            self.tcSpoolFile.SetValue(self.spoolController.seriesName)
            logger.debug('series name %s' % self.spoolController.seriesName)

            self.UpdateFreeSpace()

    def OnTcSpoolFileText(self, event):
        fn = self.tcSpoolFile.GetValue()
        if not fn == '':
            self.spoolController.seriesName = fn
        event.Skip()
        
    def OnSpoolMethodChanged(self, event):
        self.spoolController.SetSpoolMethod(self._get_spool_method())
        if self.spoolController.spoolType == 'Memory':
            self.stSpoolDirName.Hide()
            self.tcSpoolFile.Hide()
            self.stDiskSpace.Hide()
            self.bSetSpoolDir.Hide()
        else:
            self.stSpoolDirName.Show()
            self.tcSpoolFile.Show()
            self.stDiskSpace.Show()
            self.bSetSpoolDir.Show()

        if self.spoolController.spoolType in ['Cluster', 'File']:
            self._comp_p.Show()
        else:
            self._comp_p.Hide()

        self._pan_spool_to.cascading_layout()

        self.stSpoolDirName.SetLabel(self.spoolController.display_dirname)
        self.tcSpoolFile.SetValue(self.spoolController.seriesName)

        self.UpdateFreeSpace()



        
