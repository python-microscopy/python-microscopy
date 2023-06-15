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
] = [wx.NewIdRef() for _init_ctrls in range(24)]



class AndorNeoPanel(wx.Panel):


    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_ANDORFRAME,
              parent=prnt, size=wx.Size(-1, -1))
        #self.SetClientSize(wx.Size(244, 327))

        vsizer=wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        sbCooling = wx.StaticBoxSizer(wx.StaticBox(self, -1, u'Cooling [\N{DEGREE SIGN}C]'), wx.VERTICAL)
        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.tCCDTemp = wx.TextCtrl(self, -1, '0', size=(30, -1))
        hsizer2.Add(self.tCCDTemp, 1, wx.wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        sbCooling.Add(hsizer2, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.bSetTemp = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetTemp.Bind(wx.EVT_BUTTON, self.OnBSetTempButton)
        hsizer2.Add(self.bSetTemp, 0, wx.ALIGN_CENTER_VERTICAL, 0)

        hsizer.Add(sbCooling, 1, wx.EXPAND|wx.RIGHT, 5)


        sbEMGain = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'EM Gain', size=(30, -1)), wx.VERTICAL)
        hsizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.tEMGain = wx.ComboBox(self, -1, '0', choices=['0', '%d' % self.cam.DefaultEMGain], size=[30,-1], style=wx.CB_DROPDOWN|wx.TE_PROCESS_ENTER)
        self.tEMGain.Bind(wx.EVT_TEXT, self.OnEMGainTextChange)
        self.tEMGain.Bind(wx.EVT_TEXT_ENTER, self.OnBSetGainButton)
        self.tEMGain.Bind(wx.EVT_COMBOBOX, self.OnBSetGainButton)
        hsizer2.Add(self.tEMGain, 1, wx.EXPAND|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        self.bSetGain = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        self.bSetGain.Bind(wx.EVT_BUTTON, self.OnBSetGainButton)
        hsizer2.Add(self.bSetGain, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 0)

        sbEMGain.Add(hsizer2, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.stTrueEMGain = wx.StaticText(self, -1, '????')
        self.stTrueEMGain.SetForegroundColour(wx.RED)
        sbEMGain.Add(self.stTrueEMGain, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        hsizer.Add(sbEMGain, 1, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.cbShutter = wx.CheckBox(self, -1, u'Spurious Noise Filter')
        self.cbShutter.SetValue(False)
        self.cbShutter.Bind(wx.EVT_CHECKBOX, self.OnCbShutterCheckbox)
        vsizer.Add(self.cbShutter, 0, wx.ALIGN_CENTER_VERTICAL|wx.TOP, 5)

        self.cbStaticBlemishCorrection = wx.CheckBox(self, -1, u'Static Blemish Correction')
        self.cbStaticBlemishCorrection.SetValue(False)
        self.cbStaticBlemishCorrection.Bind(wx.EVT_CHECKBOX, self.OnCbStaticBlemishCorrection)
        vsizer.Add(self.cbStaticBlemishCorrection, 0, wx.ALIGN_CENTER_VERTICAL|wx.TOP, 5)

        self.SetSizer(vsizer)

        

    def __init__(self, parent, cam, scope):
        self.cam = cam
        self.scope = scope

        self._init_ctrls(parent)

        self.tCCDTemp.ChangeValue(repr(self.cam.GetCCDTempSetPoint()))
        self.tEMGain.SetValue(repr(self.cam.GetEMGain()))

        self.OnEMGainTextChange(None)

    def OnBSetTempButton(self, event):
        self.scope.pa.stop()
        self.cam.SetCCDTemp(float(self.tCCDTemp.GetValue()))
        self.OnEMGainTextChange(None)
        self.scope.pa.start()

    def OnBSetGainButton(self, event):
        self.scope.pa.stop()
        self.cam.SetEMGain(int(self.tEMGain.GetValue()))
        self.scope.pa.start()

    def OnCbShutterCheckbox(self, event):
        self.scope.pa.stop()
        self.cam.SpuriousNoiseFilter.setValue(self.cbShutter.GetValue())
        self.scope.pa.start()
        #event.Skip()

    def OnCbStaticBlemishCorrection(self, event):
        self.scope.pa.stop()
        self.cam.StaticBlemishCorrection.setValue(self.cbStaticBlemishCorrection.GetValue())
        self.scope.pa.start()

    def OnEMGainTextChange(self, event):
        calEMGain = ccdCalibrator.getCalibratedCCDGain(float(self.tEMGain.GetValue()), self.cam.GetCCDTempSetPoint())
        if calEMGain == None:
            self.stTrueEMGain.SetLabel('True Gain = ????')
            self.stTrueEMGain.SetForegroundColour(wx.RED)
        else:
            self.stTrueEMGain.SetLabel('True Gain = %3.2f' % calEMGain)
            self.stTrueEMGain.SetForegroundColour(wx.BLUE)

    def refresh(self):
        self.cbShutter.SetValue(self.cam.SpuriousNoiseFilter.getValue())
        self.cbStaticBlemishCorrection.SetValue(self.cam.StaticBlemishCorrection.getValue())
