#!/usr/bin/python

##################
# AndorControlFrame.py
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

#Boa:Frame:AndorFrame



import wx
from PYME.Acquire.Hardware import ccdCalibrator
from PYME.ui.autoFoldPanel import collapsingPane



def create(parent):

    return AndorFrame(parent)



[wxID_ANDORFRAME, wxID_ANDORFRAMEBSETGAIN, wxID_ANDORFRAMEBSETTEMP, 
 wxID_ANDORFRAMEBUPDATEINT, wxID_ANDORFRAMECBBASELINECLAMP, 
 wxID_ANDORFRAMECBFRAMETRANSFER, wxID_ANDORFRAMECBSHUTTER, 
 wxID_ANDORFRAMECHHORIZCLOCK, wxID_ANDORFRAMECHVERTCLOCK, 
 wxID_ANDORFRAMEPANEL1, wxID_ANDORFRAMERBCONTIN, wxID_ANDORFRAMERBSINGLESHOT, 
 wxID_ANDORFRAMESTATICBOX1, wxID_ANDORFRAMESTATICBOX2, 
 wxID_ANDORFRAMESTATICBOX3, wxID_ANDORFRAMESTATICBOX4, 
 wxID_ANDORFRAMESTATICTEXT1, wxID_ANDORFRAMESTATICTEXT2, 
 wxID_ANDORFRAMESTATICTEXT3, wxID_ANDORFRAMESTATICTEXT4, 
 wxID_ANDORFRAMESTATICTEXT5, wxID_ANDORFRAMESTATICTEXT6, 
 wxID_ANDORFRAMETCCDTEMP, wxID_ANDORFRAMETEMGAIN, 
] = [wx.NewId() for _init_ctrls in range(24)]


from PYME.ui import manualFoldPanel as afp

class AndorPanel(afp.foldingPane):

    def _createCollapsingPane(self):
        clp = afp.collapsingPane(self, caption='Advanced ...')#|wx.CP_NO_TLW_RESIZE)
        #clp = wx.CollapsiblePane(self, label='Advanced ...', style = wx.CP_DEFAULT_STYLE)#|wx.CP_NO_TLW_RESIZE)

        #clp.Expand()
        #cp = clp.GetPane()

        cp = wx.Panel(clp, -1)
        #cp = clp = wx.Panel(self, -1)

        vsizer=wx.BoxSizer(wx.VERTICAL)

        sbAqMode = wx.StaticBoxSizer(wx.StaticBox(cp, -1, 'Acquisition Mode'), wx.HORIZONTAL)

        self.rbSingleShot = wx.RadioButton(cp, -1, 'Single Shot')
        self.rbSingleShot.SetValue(False)
        #self.rbSingleShot.SetToolTipString('Allows multiple channels with different integration times and good shutter synchronisation')
        self.rbSingleShot.Bind(wx.EVT_RADIOBUTTON,self.OnRbSingleShotRadiobutton)
        sbAqMode.Add(self.rbSingleShot, 1, wx.EXPAND, 0)

        self.rbContin = wx.RadioButton(cp, -1, 'Continuous')
        self.rbContin.SetValue(True)
        #self.rbContin.SetToolTipString('Allows fastest speeds, albeit without good syncronisation (fixable) or integration time flexibility')
        self.rbContin.Bind(wx.EVT_RADIOBUTTON, self.OnRbContinRadiobutton)
        sbAqMode.Add(self.rbContin, 1, wx.EXPAND, 0)

        vsizer.Add(sbAqMode, 0, wx.EXPAND|wx.TOP, 5)

        #self.bUpdateInt = wx.Button(id=wxID_ANDORFRAMEBUPDATEINT,
        #      label='Update Integration Time', name='bUpdateInt',
        #      parent=self, pos=wx.Point(104, 147), size=wx.Size(128, 23),
        #      style=0)
        #self.bUpdateInt.Enable(False)
        #self.bUpdateInt.Bind(wx.EVT_BUTTON, self.OnBUpdateIntButton,
        #      id=wxID_ANDORFRAMEBUPDATEINT)

        sbReadout = wx.StaticBoxSizer(wx.StaticBox(cp, -1, 'Readout Settings'), wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(cp, -1, 'Horizontal Clock:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)

        self.chHorizClock = wx.Choice(cp, -1, choices=[])
        self.chHorizClock.Bind(wx.EVT_CHOICE, self.OnChHorizClockChoice)
        hsizer.Add(self.chHorizClock, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)

        hsizer.Add(wx.StaticText(cp, -1, 'MHz'), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sbReadout.Add(hsizer, 0, 0, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(cp, -1, 'Vertical Clock:'), 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)

        self.chVertClock = wx.Choice(cp, -1, choices=[])
        self.chVertClock.Bind(wx.EVT_CHOICE, self.OnChVertClockChoice)
        hsizer.Add(self.chVertClock, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)

        hsizer.Add(wx.StaticText(cp, -1, u'\u03BCs'), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sbReadout.Add(hsizer, 0, 0, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.cbFrameTransfer = wx.ToggleButton(cp, -1, u'Frame Transfer')
        self.cbFrameTransfer.SetValue(True)
        self.cbFrameTransfer.Bind(wx.EVT_TOGGLEBUTTON, self.OnCbFrameTransferCheckbox)
        hsizer.Add(self.cbFrameTransfer, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)

        self.cbBaselineClamp = wx.ToggleButton(cp, -1, u'Baseline Clamp')
        self.cbBaselineClamp.SetValue(self.cam.GetBaselineClamp())
        self.cbBaselineClamp.Bind(wx.EVT_TOGGLEBUTTON, self.OnCbBaselineClampCheckbox)
        hsizer.Add(self.cbBaselineClamp, 0, wx.ALIGN_CENTER_VERTICAL, 0)

        sbReadout.Add(hsizer, 0, wx.TOP, 2)
        vsizer.Add(sbReadout, 0, wx.EXPAND|wx.TOP, 5)

        cp.SetSizer(vsizer)

        clp.AddNewElement(cp)

        return clp

    def _init_ctrls(self):
        # generated method, don't edit
        
        pan = wx.Panel(parent=self, id=wxID_ANDORFRAME,size=wx.Size(-1, -1))

        vsizer=wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        sbCooling = wx.StaticBoxSizer(wx.StaticBox(pan, -1, u'Cooling [\N{DEGREE SIGN}C]'), wx.VERTICAL)
        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.tCCDTemp = wx.TextCtrl(pan, -1, '0', size=(30, -1))
        hsizer2.Add(self.tCCDTemp, 1, wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        sbCooling.Add(hsizer2, 0, wx.EXPAND, 0)

        self.bSetTemp = wx.Button(pan, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetTemp.Bind(wx.EVT_BUTTON, self.OnBSetTempButton)
        hsizer2.Add(self.bSetTemp, 0, wx.ALIGN_CENTER_VERTICAL, 0)

        hsizer.Add(sbCooling, 1, wx.EXPAND|wx.RIGHT, 5)


        sbEMGain = wx.StaticBoxSizer(wx.StaticBox(pan, -1, 'EM Gain', size=(30, -1)), wx.VERTICAL)
        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.tEMGain = wx.ComboBox(pan, -1, '0', choices=['0', '%d' % self.cam.DefaultEMGain], size=[30,-1], style=wx.CB_DROPDOWN|wx.TE_PROCESS_ENTER)
        self.tEMGain.Bind(wx.EVT_TEXT, self.OnEMGainTextChange)
        self.tEMGain.Bind(wx.EVT_TEXT_ENTER, self.OnBSetGainButton)
        self.tEMGain.Bind(wx.EVT_COMBOBOX, self.OnBSetGainButton)
        hsizer2.Add(self.tEMGain, 1, wx.EXPAND|wx.RIGHT, 5)

        self.bSetGain = wx.Button(pan, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetGain.Bind(wx.EVT_BUTTON, self.OnBSetGainButton)
        hsizer2.Add(self.bSetGain, 0, wx.EXPAND, 0)

        sbEMGain.Add(hsizer2, 0, wx.EXPAND, 0)

        self.stTrueEMGain = wx.StaticText(pan, -1, '????')
        self.stTrueEMGain.SetForegroundColour(wx.RED)
        sbEMGain.Add(self.stTrueEMGain, 0, wx.EXPAND, 0)

        hsizer.Add(sbEMGain, 1, wx.EXPAND, 0)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        self.cbShutter = wx.CheckBox(pan, -1, u'Camera Shutter Open')
        self.cbShutter.SetValue(True)
        self.cbShutter.Bind(wx.EVT_CHECKBOX, self.OnCbShutterCheckbox)
        vsizer.Add(self.cbShutter, 0, wx.TOP, 5)

        #self.cp = self._createCollapsingPane(self)

        #vsizer.Add(self.cp, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        #vsizer.AddSpacer(5)

        pan.SetSizerAndFit(vsizer)
        self.AddNewElement(pan)

        

    def __init__(self, parent, cam, scope, *args, **kwargs):
        kwargs['caption'] = kwargs.get('caption', 'EMCCD')
        afp.foldingPane.__init__(self, parent, *args, **kwargs)
        self.cam = cam
        self.scope = scope

        self._init_ctrls()
        self.AddNewElement(self._createCollapsingPane())

        self.tCCDTemp.ChangeValue(repr(self.cam.GetCCDTempSetPoint()))
        self.tEMGain.SetValue(repr(self.cam.GetEMGain()))

        self._PopulateSpeeds()
        self.OnEMGainTextChange(None)

    def OnBSetTempButton(self, event):
        self.scope.frameWrangler.stop()
        self.cam.SetCCDTemp(int(self.tCCDTemp.GetValue()))
        self.OnEMGainTextChange(None)
        self.scope.frameWrangler.start()

    def OnBSetGainButton(self, event):
        self.scope.frameWrangler.stop()
        self.cam.SetEMGain(int(self.tEMGain.GetValue()))
        self.scope.frameWrangler.start()

    def OnBStartSpoolingButton(self, event):
        #event.Skip()
        fname = wx.FileSelector('Save Images as ... (image # and .dat will be appended to filename)')
    
        if not fname is None:
            self.scope.frameWrangler.stop()
            self.cam.SpoolOn(fname)

            wx.MessageBox('Click cancel to stop spooling', 'Spooling to disk', wx.CANCEL)
            self.cam.SpoolOff()
            self.scope.frameWrangler.start()

    def OnBUpdateIntButton(self, event):
        #event.Skip()
        self.scope.frameWrangler.stop()
        self.scope.frameWrangler.start()

    def OnRbSingleShotRadiobutton(self, event):
        #event.Skip()
        if self.cam.contMode:
            self.scope.frameWrangler.stop()
            self.cam.SetAcquisitionMode(self.cam.MODE_SINGLE_SHOT)
            #self.bUpdateInt.Enable(False)
            self.scope.frameWrangler.start()

    def OnRbContinRadiobutton(self, event):
        #event.Skip()
        if not self.cam.contMode:
            self.scope.frameWrangler.stop()
            self.cam.SetAcquisitionMode(self.cam.MODE_CONTINUOUS)
            #self.bUpdateInt.Enable(True)
            self.scope.frameWrangler.start()

    def OnChHorizClockChoice(self, event):
        #event.Skip()
        self.scope.frameWrangler.stop()
        self.cam.SetHorizShiftSpeed(self.chHorizClock.GetSelection())
        self.scope.frameWrangler.start()

    def OnChVertClockChoice(self, event):
        #event.Skip()
        self.scope.frameWrangler.stop()
        self.cam.SetVerticalShiftSpeed(self.chVertClock.GetSelection())
        self.scope.frameWrangler.start()

    def OnCbFrameTransferCheckbox(self, event):
        #event.Skip()
        self.scope.frameWrangler.stop()
        self.cam.SetFrameTransfer(self.cbFrameTransfer.GetValue())
        self.scope.frameWrangler.start()

    def _PopulateSpeeds(self):
        for hs in self.cam.HorizShiftSpeeds[0][0]:
            self.chHorizClock.Append('%3.0f' % hs)

        self.chHorizClock.SetSelection(self.cam.HSSpeed)            

        for i in range(len(self.cam.vertShiftSpeeds)):
            if i < self.cam.fastestRecVSInd:
                self.chVertClock.Append('[%2.2f]' % self.cam.vertShiftSpeeds[i])
            else:
                self.chVertClock.Append('%2.2f' % self.cam.vertShiftSpeeds[i])            

        self.chVertClock.SetSelection(self.cam.VSSpeed)
        self.cbFrameTransfer.SetValue(self.cam.frameTransferMode)

    def OnCbShutterCheckbox(self, event):
        self.scope.frameWrangler.stop()
        self.cam.SetShutter(self.cbShutter.GetValue())
        self.scope.frameWrangler.start()
        #event.Skip()

    def OnCbBaselineClampCheckbox(self, event):
        #event.Skip()
        self.scope.frameWrangler.stop()
        self.cam.SetBaselineClamp(self.cbBaselineClamp.GetValue())
        self.scope.frameWrangler.start()

    def OnEMGainTextChange(self, event):
        calEMGain = ccdCalibrator.getCalibratedCCDGain(float(self.tEMGain.GetValue()), self.cam.GetCCDTempSetPoint())
        if calEMGain is None:
            self.stTrueEMGain.SetLabel('True Gain = ????')
            self.stTrueEMGain.SetForegroundColour(wx.RED)
        else:
            self.stTrueEMGain.SetLabel('True Gain = %3.2f' % calEMGain)
            self.stTrueEMGain.SetForegroundColour(wx.BLUE)



            
