#Boa:Frame:AndorFrame



import wx



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



class AndorPanel(wx.Panel):

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_ANDORFRAME,
              parent=prnt, size=wx.Size(252, 327))
        #self.SetClientSize(wx.Size(244, 327))

        self.panel1 = wx.Panel(id=wxID_ANDORFRAMEPANEL1, name='panel1',
              parent=self, pos=wx.Point(0, 0), size=wx.Size(244, 327),
              style=wx.TAB_TRAVERSAL)

        self.staticBox1 = wx.StaticBox(id=wxID_ANDORFRAMESTATICBOX1,
              label='Cooling', name='staticBox1', parent=self.panel1,
              pos=wx.Point(8, 10), size=wx.Size(121, 86), style=0)

        self.staticBox2 = wx.StaticBox(id=wxID_ANDORFRAMESTATICBOX2,
              label='EM Gain', name='staticBox2', parent=self.panel1,
              pos=wx.Point(145, 10), size=wx.Size(96, 86), style=0)

        self.tCCDTemp = wx.TextCtrl(id=wxID_ANDORFRAMETCCDTEMP, name='tCCDTemp',
              parent=self.panel1, pos=wx.Point(21, 31), size=wx.Size(56, 21),
              style=0, value='0')

        self.staticText1 = wx.StaticText(id=wxID_ANDORFRAMESTATICTEXT1,
              label='deg C', name='staticText1', parent=self.panel1,
              pos=wx.Point(87, 34), size=wx.Size(28, 13), style=0)

        self.bSetTemp = wx.Button(id=wxID_ANDORFRAMEBSETTEMP, label='Set',
              name='bSetTemp', parent=self.panel1, pos=wx.Point(31, 62),
              size=wx.Size(75, 23), style=0)
        self.bSetTemp.Bind(wx.EVT_BUTTON, self.OnBSetTempButton,
              id=wxID_ANDORFRAMEBSETTEMP)

        self.tEMGain = wx.TextCtrl(id=wxID_ANDORFRAMETEMGAIN, name='tEMGain',
              parent=self.panel1, pos=wx.Point(164, 31), size=wx.Size(56, 21),
              style=0, value='textCtrl1')

        self.bSetGain = wx.Button(id=wxID_ANDORFRAMEBSETGAIN, label='Set',
              name='bSetGain', parent=self.panel1, pos=wx.Point(156, 62),
              size=wx.Size(75, 23), style=0)
        self.bSetGain.Bind(wx.EVT_BUTTON, self.OnBSetGainButton,
              id=wxID_ANDORFRAMEBSETGAIN)

        self.staticBox3 = wx.StaticBox(id=wxID_ANDORFRAMESTATICBOX3,
              label='Acquisition Mode', name='staticBox3', parent=self.panel1,
              pos=wx.Point(8, 103), size=wx.Size(232, 80), style=0)

        self.rbSingleShot = wx.RadioButton(id=wxID_ANDORFRAMERBSINGLESHOT,
              label='Single Shot', name='rbSingleShot', parent=self.panel1,
              pos=wx.Point(24, 127), size=wx.Size(81, 13), style=0)
        self.rbSingleShot.SetValue(True)
        self.rbSingleShot.SetToolTipString('Allows multiple channels with different integration times and good shutter synchronisation')
        self.rbSingleShot.Bind(wx.EVT_RADIOBUTTON,
              self.OnRbSingleShotRadiobutton, id=wxID_ANDORFRAMERBSINGLESHOT)

        self.rbContin = wx.RadioButton(id=wxID_ANDORFRAMERBCONTIN,
              label='Continuous', name='rbContin', parent=self.panel1,
              pos=wx.Point(24, 151), size=wx.Size(81, 13), style=0)
        self.rbContin.SetValue(False)
        self.rbContin.SetToolTipString('Allows fastest speeds, albeit without good syncronisation (fixable) or integration time flexibility')
        self.rbContin.Bind(wx.EVT_RADIOBUTTON, self.OnRbContinRadiobutton,
              id=wxID_ANDORFRAMERBCONTIN)

        self.bUpdateInt = wx.Button(id=wxID_ANDORFRAMEBUPDATEINT,
              label='Update Integration Time', name='bUpdateInt',
              parent=self.panel1, pos=wx.Point(104, 147), size=wx.Size(128, 23),
              style=0)
        self.bUpdateInt.Enable(False)
        self.bUpdateInt.Bind(wx.EVT_BUTTON, self.OnBUpdateIntButton,
              id=wxID_ANDORFRAMEBUPDATEINT)

        self.staticBox4 = wx.StaticBox(id=wxID_ANDORFRAMESTATICBOX4,
              label='Readout Settings', name='staticBox4', parent=self.panel1,
              pos=wx.Point(8, 192), size=wx.Size(232, 104), style=0)

        self.staticText2 = wx.StaticText(id=wxID_ANDORFRAMESTATICTEXT2,
              label='Horizontal Clock:', name='staticText2', parent=self.panel1,
              pos=wx.Point(24, 216), size=wx.Size(80, 16), style=0)

        self.staticText3 = wx.StaticText(id=wxID_ANDORFRAMESTATICTEXT3,
              label='Horizontal Clock:', name='staticText3', parent=self.panel1,
              pos=wx.Point(24, 216), size=wx.Size(80, 16), style=0)

        self.staticText4 = wx.StaticText(id=wxID_ANDORFRAMESTATICTEXT4,
              label='Vertical Clock:', name='staticText4', parent=self.panel1,
              pos=wx.Point(24, 243), size=wx.Size(68, 13), style=0)

        self.chHorizClock = wx.Choice(choices=[],
              id=wxID_ANDORFRAMECHHORIZCLOCK, name='chHorizClock',
              parent=self.panel1, pos=wx.Point(115, 213), size=wx.Size(77, 21),
              style=0)
        self.chHorizClock.Bind(wx.EVT_CHOICE, self.OnChHorizClockChoice,
              id=wxID_ANDORFRAMECHHORIZCLOCK)

        self.staticText5 = wx.StaticText(id=wxID_ANDORFRAMESTATICTEXT5,
              label='MHz', name='staticText5', parent=self.panel1,
              pos=wx.Point(199, 216), size=wx.Size(20, 13), style=0)

        self.chVertClock = wx.Choice(choices=[], id=wxID_ANDORFRAMECHVERTCLOCK,
              name='chVertClock', parent=self.panel1, pos=wx.Point(115, 240),
              size=wx.Size(77, 21), style=0)
        self.chVertClock.Bind(wx.EVT_CHOICE, self.OnChVertClockChoice,
              id=wxID_ANDORFRAMECHVERTCLOCK)

        self.staticText6 = wx.StaticText(id=wxID_ANDORFRAMESTATICTEXT6,
              label='us', name='staticText6', parent=self.panel1,
              pos=wx.Point(199, 243), size=wx.Size(11, 13), style=0)

        self.cbFrameTransfer = wx.CheckBox(id=wxID_ANDORFRAMECBFRAMETRANSFER,
              label=u'Frame Transfer', name='cbFrameTransfer',
              parent=self.panel1, pos=wx.Point(24, 272), size=wx.Size(96, 13),
              style=0)
        self.cbFrameTransfer.SetValue(False)
        self.cbFrameTransfer.Bind(wx.EVT_CHECKBOX,
              self.OnCbFrameTransferCheckbox,
              id=wxID_ANDORFRAMECBFRAMETRANSFER)

        self.cbShutter = wx.CheckBox(id=wxID_ANDORFRAMECBSHUTTER,
              label=u'Camera Shutter Open', name=u'cbShutter',
              parent=self.panel1, pos=wx.Point(16, 304), size=wx.Size(160, 13),
              style=0)
        self.cbShutter.SetValue(True)
        self.cbShutter.Bind(wx.EVT_CHECKBOX, self.OnCbShutterCheckbox,
              id=wxID_ANDORFRAMECBSHUTTER)

        self.cbBaselineClamp = wx.CheckBox(id=wxID_ANDORFRAMECBBASELINECLAMP,
              label=u'Baseline Clamp', name=u'cbBaselineClamp',
              parent=self.panel1, pos=wx.Point(136, 272), size=wx.Size(96, 13),
              style=0)
        self.cbBaselineClamp.SetValue(False)
        self.cbBaselineClamp.Bind(wx.EVT_CHECKBOX,
              self.OnCbBaselineClampCheckbox,
              id=wxID_ANDORFRAMECBBASELINECLAMP)

    def __init__(self, parent, cam, scope):

        self._init_ctrls(parent)

        

        self.cam = cam

        self.scope = scope

        

        self.tCCDTemp.ChangeValue(repr(self.cam.GetCCDTempSetPoint()))

        self.tEMGain.ChangeValue(repr(self.cam.GetEMGain()))

        

        self._PopulateSpeeds()



    def OnBSetTempButton(self, event):

        self.scope.pa.stop()

        self.cam.SetCCDTemp(int(self.tCCDTemp.GetValue()))

        self.scope.pa.start()



    def OnBSetGainButton(self, event):

        self.scope.pa.stop()

        self.cam.SetEMGain(int(self.tEMGain.GetValue()))

        self.scope.pa.start()



    def OnBStartSpoolingButton(self, event):

        #event.Skip()

        fname = wx.FileSelector('Save Images as ... (image # and .dat will be appended to filename)')

        

        if not fname == None:

            self.scope.pa.stop()

        

            self.cam.SpoolOn(fname)

            

            wx.MessageBox('Click cancel to stop spooling', 'Spooling to disk', wx.CANCEL)

            

            self.cam.SpoolOff()

            

            self.scope.pa.start()



    def OnBUpdateIntButton(self, event):

        #event.Skip()

        self.scope.pa.stop()

        self.scope.pa.start()



    def OnRbSingleShotRadiobutton(self, event):

        #event.Skip()

        if self.cam.contMode:

            self.scope.pa.stop()

            self.cam.SetAcquisitionMode(self.cam.MODE_SINGLE_SHOT)

            self.bUpdateInt.Enable(False)

            self.scope.pa.start()



    def OnRbContinRadiobutton(self, event):

        #event.Skip()

        if not self.cam.contMode:

            self.scope.pa.stop()

            self.cam.SetAcquisitionMode(self.cam.MODE_CONTINUOUS)

            self.bUpdateInt.Enable(True)

            self.scope.pa.start()



    def OnChHorizClockChoice(self, event):

        #event.Skip()

        self.scope.pa.stop()

        self.cam.SetHorizShiftSpeed(self.chHorizClock.GetSelection())

        self.scope.pa.start()



    def OnChVertClockChoice(self, event):

        #event.Skip()

        self.scope.pa.stop()

        self.cam.SetVerticalShiftSpeed(self.chVertClock.GetSelection())

        self.scope.pa.start()



    def OnCbFrameTransferCheckbox(self, event):

        #event.Skip()

        self.scope.pa.stop()

        self.cam.SetFrameTransfer(self.cbFrameTransfer.GetValue())

        self.scope.pa.start()



    def _PopulateSpeeds(self):

        for hs in self.cam.HorizShiftSpeeds[0][0]:

            self.chHorizClock.Append('%f' % hs)

            

        self.chHorizClock.SetSelection(self.cam.HSSpeed)

            

        for i in range(len(self.cam.vertShiftSpeeds)):

            if i < self.cam.fastestRecVSInd:

                self.chVertClock.Append('[%2.2f]' % self.cam.vertShiftSpeeds[i])

            else:

                self.chVertClock.Append('%2.2f' % self.cam.vertShiftSpeeds[i])

                

        self.chVertClock.SetSelection(self.cam.VSSpeed)

        

        self.cbFrameTransfer.SetValue(self.cam.frameTransferMode)

    def OnCbShutterCheckbox(self, event):
        self.scope.pa.stop()

        self.cam.SetShutter(self.cbShutter.GetValue())

        self.scope.pa.start()
        
        #event.Skip()

    def OnCbBaselineClampCheckbox(self, event):
        #event.Skip()
        self.scope.pa.stop()

        self.cam.SetBaselineClamp(self.cbShutter.GetValue())

        self.scope.pa.start()

            
