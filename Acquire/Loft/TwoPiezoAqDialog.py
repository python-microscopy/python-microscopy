#!/usr/bin/python

##################
# TwoPiezoAqDialog.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:Dialog:TwoPiezoAqDialog

import wx
import two_piezo_aq

def create(parent):
    return TwoPiezoAqDialog(parent)

[wxID_TWOPIEZOAQDIALOG, wxID_TWOPIEZOAQDIALOGBCALCPH1, 
 wxID_TWOPIEZOAQDIALOGBCALCSTEP1, wxID_TWOPIEZOAQDIALOGBEND0HERE, 
 wxID_TWOPIEZOAQDIALOGBEND1HERE, wxID_TWOPIEZOAQDIALOGBSTART0HERE, 
 wxID_TWOPIEZOAQDIALOGBSTART1HERE, wxID_TWOPIEZOAQDIALOGBSTARTAQ, 
 wxID_TWOPIEZOAQDIALOGCHPIEZO0, wxID_TWOPIEZOAQDIALOGCHPIEZO1, 
 wxID_TWOPIEZOAQDIALOGEDEND0, wxID_TWOPIEZOAQDIALOGEDEND1, 
 wxID_TWOPIEZOAQDIALOGEDLAMBDA1, wxID_TWOPIEZOAQDIALOGEDNUM0, 
 wxID_TWOPIEZOAQDIALOGEDNUM1, wxID_TWOPIEZOAQDIALOGEDPHASEMULT1, 
 wxID_TWOPIEZOAQDIALOGEDPHASESTEP1, wxID_TWOPIEZOAQDIALOGEDSTART0, 
 wxID_TWOPIEZOAQDIALOGEDSTART1, wxID_TWOPIEZOAQDIALOGEDSTEP0, 
 wxID_TWOPIEZOAQDIALOGEDSTEP1, wxID_TWOPIEZOAQDIALOGRBMODE0, 
 wxID_TWOPIEZOAQDIALOGRBMODE1, wxID_TWOPIEZOAQDIALOGSTATICBOX1, 
 wxID_TWOPIEZOAQDIALOGSTATICBOX2, wxID_TWOPIEZOAQDIALOGSTATICBOX5, 
 wxID_TWOPIEZOAQDIALOGSTATICTEXT1, wxID_TWOPIEZOAQDIALOGSTATICTEXT10, 
 wxID_TWOPIEZOAQDIALOGSTATICTEXT11, wxID_TWOPIEZOAQDIALOGSTATICTEXT12, 
 wxID_TWOPIEZOAQDIALOGSTATICTEXT13, wxID_TWOPIEZOAQDIALOGSTATICTEXT14, 
 wxID_TWOPIEZOAQDIALOGSTATICTEXT2, wxID_TWOPIEZOAQDIALOGSTATICTEXT3, 
 wxID_TWOPIEZOAQDIALOGSTATICTEXT4, wxID_TWOPIEZOAQDIALOGSTATICTEXT5, 
 wxID_TWOPIEZOAQDIALOGSTATICTEXT6, wxID_TWOPIEZOAQDIALOGSTATICTEXT7, 
 wxID_TWOPIEZOAQDIALOGSTATICTEXT8, wxID_TWOPIEZOAQDIALOGSTATICTEXT9, 
] = [wx.NewId() for _init_ctrls in range(40)]

class TwoPiezoAqDialog(wx.Dialog):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Dialog.__init__(self, id=wxID_TWOPIEZOAQDIALOG,
              name='TwoPiezoAqDialog', parent=prnt, pos=wx.Point(386, 228),
              size=wx.Size(463, 356), style=wx.DEFAULT_DIALOG_STYLE,
              title='Two Piezo Sequence')
        self.SetClientSize(wx.Size(455, 329))

        self.staticBox2 = wx.StaticBox(id=wxID_TWOPIEZOAQDIALOGSTATICBOX2,
              label='Axis 1 (normally Z)', name='staticBox2', parent=self,
              pos=wx.Point(16, 12), size=wx.Size(192, 228), style=0)

        self.chPiezo0 = wx.Choice(choices=[], id=wxID_TWOPIEZOAQDIALOGCHPIEZO0,
              name='chPiezo0', parent=self, pos=wx.Point(79, 33),
              size=wx.Size(98, 21), style=0)
        self.chPiezo0.Bind(wx.EVT_CHOICE, self.OnChPiezo0Choice,
              id=wxID_TWOPIEZOAQDIALOGCHPIEZO0)

        self.staticText1 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT1,
              label='Piezo:', name='staticText1', parent=self, pos=wx.Point(33,
              35), size=wx.Size(29, 13), style=0)

        self.edStep0 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDSTEP0,
              name='edStep0', parent=self, pos=wx.Point(121, 62),
              size=wx.Size(56, 21), style=0, value='0.04')
        self.edStep0.Bind(wx.EVT_KILL_FOCUS, self.OnEdStep0KillFocus)

        self.staticText2 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT2,
              label='Step size [um]:', name='staticText2', parent=self,
              pos=wx.Point(35, 64), size=wx.Size(72, 13), style=0)

        self.edEnd0 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDEND0, name='edEnd0',
              parent=self, pos=wx.Point(105, 179), size=wx.Size(56, 21),
              style=0, value='0')
        self.edEnd0.Bind(wx.EVT_TEXT_ENTER, self.OnEdEnd0TextEnter,
              id=wxID_TWOPIEZOAQDIALOGEDEND0)
        self.edEnd0.Bind(wx.EVT_KILL_FOCUS, self.OnEdEnd0KillFocus)

        self.edNum0 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDNUM0, name='edNum0',
              parent=self, pos=wx.Point(105, 203), size=wx.Size(56, 21),
              style=0, value='100')
        self.edNum0.Bind(wx.EVT_TEXT_ENTER, self.OnEdNum0TextEnter,
              id=wxID_TWOPIEZOAQDIALOGEDNUM0)
        self.edNum0.Bind(wx.EVT_KILL_FOCUS, self.OnEdNum0KillFocus)

        self.edStart0 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDSTART0,
              name='edStart0', parent=self, pos=wx.Point(105, 155),
              size=wx.Size(56, 21), style=0, value='0')
        self.edStart0.Bind(wx.EVT_TEXT_ENTER, self.OnEdStart0TextEnter,
              id=wxID_TWOPIEZOAQDIALOGEDSTART0)
        self.edStart0.Bind(wx.EVT_KILL_FOCUS, self.OnEdStart0KillFocus)

        self.staticText3 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT3,
              label='Start Pos [um]:', name='staticText3', parent=self,
              pos=wx.Point(27, 158), size=wx.Size(73, 13), style=0)

        self.staticText4 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT4,
              label='End Pos [um]:', name='staticText4', parent=self,
              pos=wx.Point(33, 181), size=wx.Size(67, 13), style=0)

        self.staticText5 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT5,
              label='Number Slices:', name='staticText5', parent=self,
              pos=wx.Point(29, 204), size=wx.Size(70, 13), style=0)

        self.bStart0Here = wx.Button(id=wxID_TWOPIEZOAQDIALOGBSTART0HERE,
              label='here', name='bStart0Here', parent=self, pos=wx.Point(168,
              155), size=wx.Size(32, 23), style=0)
        self.bStart0Here.Bind(wx.EVT_BUTTON, self.OnBStart0HereButton,
              id=wxID_TWOPIEZOAQDIALOGBSTART0HERE)

        self.bEnd0Here = wx.Button(id=wxID_TWOPIEZOAQDIALOGBEND0HERE,
              label='here', name='bEnd0Here', parent=self, pos=wx.Point(168,
              179), size=wx.Size(32, 23), style=0)
        self.bEnd0Here.Bind(wx.EVT_BUTTON, self.OnBEnd0HereButton,
              id=wxID_TWOPIEZOAQDIALOGBEND0HERE)

        self.staticText7 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT7,
              label='Number Slices:', name='staticText7', parent=self,
              pos=wx.Point(262, 284), size=wx.Size(70, 13), style=0)

        self.edNum1 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDNUM1, name='edNum1',
              parent=self, pos=wx.Point(338, 283), size=wx.Size(56, 21),
              style=0, value='5')
        self.edNum1.Bind(wx.EVT_KILL_FOCUS, self.OnEdNum1KillFocus)

        self.edEnd1 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDEND1, name='edEnd1',
              parent=self, pos=wx.Point(338, 259), size=wx.Size(56, 21),
              style=0, value='0')
        self.edEnd1.Bind(wx.EVT_KILL_FOCUS, self.OnEdEnd1KillFocus)

        self.staticText6 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT6,
              label='End Pos [um]:', name='staticText6', parent=self,
              pos=wx.Point(266, 261), size=wx.Size(67, 13), style=0)

        self.staticText9 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT9,
              label='Start Pos [um]:', name='staticText9', parent=self,
              pos=wx.Point(260, 238), size=wx.Size(73, 13), style=0)

        self.edStart1 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDSTART1,
              name='edStart1', parent=self, pos=wx.Point(338, 235),
              size=wx.Size(56, 21), style=0, value='0')
        self.edStart1.Bind(wx.EVT_KILL_FOCUS, self.OnEdStart1KillFocus)

        self.bStart1Here = wx.Button(id=wxID_TWOPIEZOAQDIALOGBSTART1HERE,
              label='here', name='bStart1Here', parent=self, pos=wx.Point(401,
              235), size=wx.Size(32, 23), style=0)
        self.bStart1Here.Bind(wx.EVT_BUTTON, self.OnBStart1HereButton,
              id=wxID_TWOPIEZOAQDIALOGBSTART1HERE)

        self.bEnd1Here = wx.Button(id=wxID_TWOPIEZOAQDIALOGBEND1HERE,
              label='here', name='bEnd1Here', parent=self, pos=wx.Point(401,
              259), size=wx.Size(32, 23), style=0)
        self.bEnd1Here.Bind(wx.EVT_BUTTON, self.OnBEnd1HereButton,
              id=wxID_TWOPIEZOAQDIALOGBEND1HERE)

        self.staticText8 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT8,
              label='um', name='staticText8', parent=self, pos=wx.Point(308,
              74), size=wx.Size(14, 13), style=0)

        self.edStep1 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDSTEP1,
              name='edStep1', parent=self, pos=wx.Point(264, 71),
              size=wx.Size(39, 21), style=0, value='0.04')
        self.edStep1.Bind(wx.EVT_KILL_FOCUS, self.OnEdStep1KillFocus)

        self.chPiezo1 = wx.Choice(choices=[], id=wxID_TWOPIEZOAQDIALOGCHPIEZO1,
              name='chPiezo1', parent=self, pos=wx.Point(312, 30),
              size=wx.Size(98, 21), style=0)
        self.chPiezo1.Bind(wx.EVT_CHOICE, self.OnChPiezo1Choice,
              id=wxID_TWOPIEZOAQDIALOGCHPIEZO1)

        self.staticText10 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT10,
              label='Piezo:', name='staticText10', parent=self,
              pos=wx.Point(266, 32), size=wx.Size(29, 13), style=0)

        self.staticBox1 = wx.StaticBox(id=wxID_TWOPIEZOAQDIALOGSTATICBOX1,
              label='Axis 2 (normally phase)', name='staticBox1', parent=self,
              pos=wx.Point(248, 10), size=wx.Size(192, 302), style=0)

        self.staticBox5 = wx.StaticBox(id=wxID_TWOPIEZOAQDIALOGSTATICBOX5,
              label='Step Size', name='staticBox5', parent=self,
              pos=wx.Point(256, 53), size=wx.Size(176, 104), style=0)

        self.staticText11 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT11,
              label='x pi', name='staticText11', parent=self, pos=wx.Point(407,
              73), size=wx.Size(17, 13), style=0)

        self.edPhaseStep1 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDPHASESTEP1,
              name='edPhaseStep1', parent=self, pos=wx.Point(365, 71),
              size=wx.Size(39, 21), style=0, value='0.66')
        self.edPhaseStep1.Bind(wx.EVT_KILL_FOCUS, self.OnEdPhaseStep1KillFocus)

        self.bCalcStep1 = wx.Button(id=wxID_TWOPIEZOAQDIALOGBCALCSTEP1,
              label='<', name='bCalcStep1', parent=self, pos=wx.Point(327, 70),
              size=wx.Size(16, 23), style=0)
        self.bCalcStep1.Bind(wx.EVT_BUTTON, self.OnBCalcStep1Button,
              id=wxID_TWOPIEZOAQDIALOGBCALCSTEP1)

        self.bCalcPh1 = wx.Button(id=wxID_TWOPIEZOAQDIALOGBCALCPH1, label='>',
              name='bCalcPh1', parent=self, pos=wx.Point(345, 70),
              size=wx.Size(16, 23), style=0)
        self.bCalcPh1.Bind(wx.EVT_BUTTON, self.OnBCalcPh1Button,
              id=wxID_TWOPIEZOAQDIALOGBCALCPH1)

        self.staticText12 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT12,
              label='At Lambda = ', name='staticText12', parent=self,
              pos=wx.Point(268, 101), size=wx.Size(65, 13), style=0)

        self.edLambda1 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDLAMBDA1,
              name='edLambda1', parent=self, pos=wx.Point(340, 98),
              size=wx.Size(36, 21), style=0, value='488')
        self.edLambda1.Bind(wx.EVT_KILL_FOCUS, self.OnEdLambda1KillFocus)

        self.staticText13 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT13,
              label='nm', name='staticText13', parent=self, pos=wx.Point(380,
              101), size=wx.Size(16, 13), style=0)

        self.edPhaseMult1 = wx.TextCtrl(id=wxID_TWOPIEZOAQDIALOGEDPHASEMULT1,
              name='edPhaseMult1', parent=self, pos=wx.Point(340, 126),
              size=wx.Size(36, 21), style=0, value='2')
        self.edPhaseMult1.Bind(wx.EVT_KILL_FOCUS, self.OnEdPhaseMult1KillFocus)

        self.staticText14 = wx.StaticText(id=wxID_TWOPIEZOAQDIALOGSTATICTEXT14,
              label='with multiplier', name='staticText14', parent=self,
              pos=wx.Point(269, 128), size=wx.Size(65, 13), style=0)

        self.rbMode0 = wx.RadioBox(choices=['Start and end',
              'Middle and number'], id=wxID_TWOPIEZOAQDIALOGRBMODE0,
              label='Mode', majorDimension=1, name='rbMode0', parent=self,
              pos=wx.Point(42, 83), size=wx.Size(128, 64),
              style=wx.RA_SPECIFY_COLS)
        self.rbMode0.Bind(wx.EVT_RADIOBOX, self.OnRbMode0Radiobox,
              id=wxID_TWOPIEZOAQDIALOGRBMODE0)

        self.rbMode1 = wx.RadioBox(choices=['Start and end',
              'Middle and number'], id=wxID_TWOPIEZOAQDIALOGRBMODE1,
              label='Mode', majorDimension=1, name='rbMode1', parent=self,
              pos=wx.Point(286, 163), size=wx.Size(128, 64),
              style=wx.RA_SPECIFY_COLS)
        self.rbMode1.Bind(wx.EVT_RADIOBOX, self.OnRbMode1Radiobox,
              id=wxID_TWOPIEZOAQDIALOGRBMODE1)

        self.bStartAq = wx.Button(id=wxID_TWOPIEZOAQDIALOGBSTARTAQ,
              label='Start Aquisition', name='bStartAq', parent=self,
              pos=wx.Point(56, 264), size=wx.Size(99, 23), style=0)
        self.bStartAq.Bind(wx.EVT_BUTTON, self.OnBStartAqButton,
              id=wxID_TWOPIEZOAQDIALOGBSTARTAQ)

    def __init__(self, parent,scope):
        self.scope = scope
        self._init_ctrls(parent)
        
        if not ('tpa' in self.scope.__dict__):
            self.scope.tpa = two_piezo_aq.TwoPiezoAquisator(self.scope.chaninfo, self.scope.cam, self.scope.piezos)
            
        for pz in self.scope.piezos:
            self.chPiezo0.Append(pz[2])
            self.chPiezo1.Append(pz[2])
        self.UpdateDisp()

    def OnBStart0HereButton(self, event):
        self.scope.tpa.SetStartPos(0,self.scope.piezos[self.scope.tpa.GetScanChannel()][0].GetPos(self.scope.piezos[self.scope.tpa.GetScanChannel()][1]))
        self.UpdateDisp()

    def OnBEnd0HereButton(self, event):
        self.scope.tpa.SetEndPos(0,self.scope.piezos[self.scope.tpa.GetScanChannel()][0].GetPos(self.scope.piezos[self.scope.tpa.GetScanChannel()][1]))
        self.UpdateDisp()

    def OnBStart1HereButton(self, event):
        self.scope.tpa.SetStartPos(1,self.scope.piezos[self.scope.tpa.GetScanChannel()][0].GetPos(self.scope.piezos[self.scope.tpa.GetScanChannel()][1]))
        self.UpdateDisp()

    def OnBEnd1HereButton(self, event):
        self.scope.tpa.SetEndPos(1,self.scope.piezos[self.scope.tpa.GetScanChannel()][0].GetPos(self.scope.piezos[self.scope.tpa.GetScanChannel()][1]))
        self.UpdateDisp()

    def OnBCalcStep1Button(self, event):
        ph = float(self.edPhaseStep1.GetValue())
        lamb = float(self.edLambda1.GetValue())
        ph_mult = float(self.edPhaseMult1.GetValue())
        
        step = ph*lamb/(2*1000*ph_mult)
        
        self.scope.tpa.SetStepSize(1,step)
        self.UpdateDisp()

    def OnBCalcPh1Button(self, event):
        step = float(self.edStep1.GetValue())
        lamb = float(self.edLambda1.GetValue())
        ph_mult = float(self.edPhaseMult1.GetValue())
        
        ph = step/(lamb/(2*1000*ph_mult))
        self.edPhaseStep1.SetValue('%2.3f' % ph)

    def OnBStartAqButton(self, event):
        res = self.scope.tpa.Verify()
        if res[0]:
            self.scope.pa.stop()

            #try:
            self.scope.tpa.Prepare()
            #except:
            #    dialog = wxMessageDialog(None, 'The most likely reason is a lack of memory \nTry the following: Close any open aquisitions, Chose a ROI, Delete unnecessary channels, or decrease the # of slices', "Could not start aquisition", wx.OK)
            #    dialog.ShowModal()
            #    self.scope.tpa.ds=[]
            #    self.scope.pa.Prepare(True)
            #    self.scope.pa.start()

            self.scope.tpa.WantFrameNotification=[]
            self.scope.tpa.WantFrameNotification.append(self.scope.aq_refr)

            self.scope.tpa.WantStopNotification=[]
            self.scope.tpa.WantStopNotification.append(self.scope.aq_end)

            self.scope.tpa.start()

            self.scope.pb = wxProgressDialog('Aquisition in progress ...', 'Slice 1 of %d' % self.scope.tpa.ds.getDepth(), self.scope.tpa.ds.getDepth(), style = wxPD_APP_MODAL|wxPD_AUTO_HIDE|wxPD_REMAINING_TIME|wxPD_CAN_ABORT)              
                
            
        else:
            dialog = wxMessageDialog(None, res[2] + ' (%2.3f)'% res[3], "Parameter Error", wx.OK)
            dialog.ShowModal()
            
            #if res[1] == 'StepSize':
            #    self.tStepSize.SetFocus()
            #elif (self.scope.tpa.GetStartMode() == self.scope.tpa.CENTRE_AND_LENGTH):
            #    self.tNumSlices.SetFocus()
            #elif (res[1] == 'StartPos'):
            #    self.tStPos.SetFocus()
            #else:
            #    self.tEndPos.SetFocus() 
        #event.Skip()

    def OnRbMode0Radiobox(self, event):
        self.scope.tpa.SetStartMode(0,self.rbMode0.GetSelection())
        self.UpdateDisp()

    def OnRbMode1Radiobox(self, event):
        self.scope.tpa.SetStartMode(1,self.rbMode0.GetSelection())
        self.UpdateDisp()
        
    def UpdateDisp(self):
        self.chPiezo0.SetSelection(self.scope.tpa.GetScanChannel(0))
        self.chPiezo1.SetSelection(self.scope.tpa.GetScanChannel(1))
            
        if self.scope.tpa.GetStartMode(0) == self.scope.tpa.START_AND_END:
            self.rbMode0.SetSelection(self.scope.tpa.START_AND_END)
            
            self.edNum0.Enable(False)
            self.edStart0.Enable(True)
            self.edEnd0.Enable(True)
            self.bStart0Here.Enable(True)
            self.bEnd0Here.Enable(True)
        else:
            self.rbMode0.SetSelection(self.scope.tpa.START_AND_END)
            
            self.edNum0.Enable(True)
            self.edStart0.Enable(False)
            self.edEnd0.Enable(False)
            self.bStart0Here.Enable(False)
            self.bEnd0Here.Enable(False)
            
        if self.scope.tpa.GetStartMode(1) == self.scope.tpa.START_AND_END:
            self.rbMode1.SetSelection(self.scope.tpa.START_AND_END)
            
            self.edNum1.Enable(False)
            self.edStart1.Enable(True)
            self.edEnd1.Enable(True)
            self.bStart1Here.Enable(True)
            self.bEnd1Here.Enable(True)
        else:
            self.rbMode1.SetSelection(self.scope.tpa.START_AND_END)
            
            self.edNum1.Enable(True)
            self.edStart1.Enable(False)
            self.edEnd1.Enable(False)
            self.bStart1Here.Enable(False)
            self.bEnd1Here.Enable(False)
        
        self.edStart0.SetValue('%2.3f' % self.scope.tpa.GetStartPos(0))
        self.edEnd0.SetValue('%2.3f' % self.scope.tpa.GetEndPos(0))
        self.edStep0.SetValue('%2.3f' % self.scope.tpa.GetStepSize(0))
        self.edNum0.SetValue('%d' % self.scope.tpa.GetSeqLength(0))
        
        self.edStart1.SetValue('%2.3f' % self.scope.tpa.GetStartPos(1))
        self.edEnd1.SetValue('%2.3f' % self.scope.tpa.GetEndPos(1))
        self.edStep1.SetValue('%2.3f' % self.scope.tpa.GetStepSize(1))
        self.edNum1.SetValue('%d' % self.scope.tpa.GetSeqLength(1))

    def OnChPiezo0Choice(self, event):
        self.scope.tpa.SetScanChannel(0,self.chPiezo0.GetSelection())
        self.UpdateDisp()

    def OnEdStep0KillFocus(self, event):
        self.scope.tpa.SetStepSize(0,float(self.edStep0.GetValue()))
        self.UpdateDisp()

    def OnEdEnd0TextEnter(self, event):
        event.Skip()

    def OnEdEnd0KillFocus(self, event):
        self.scope.tpa.SetEndPos(0,float(self.edEnd0.GetValue()))
        self.UpdateDisp()

    def OnEdNum0TextEnter(self, event):
        event.Skip()

    def OnEdNum0KillFocus(self, event):
        self.scope.tpa.SetSeqLength(0, int(self.edNum0.GetValue()))
        self.UpdateDisp()

    def OnEdStart0TextEnter(self, event):
        event.Skip()

    def OnEdStart0KillFocus(self, event):
        self.scope.tpa.SetStartPos(0,float(self.edStart0.GetValue()))
        self.UpdateDisp()

    def OnEdNum1KillFocus(self, event):
        self.scope.tpa.SetSeqLength(1, int(self.edNum1.GetValue()))
        self.UpdateDisp()

    def OnEdEnd1KillFocus(self, event):
        self.scope.tpa.SetEndPos(1,float(self.edEnd1.GetValue()))
        self.UpdateDisp()

    def OnEdStart1KillFocus(self, event):
        self.scope.tpa.SetStartPos(1,float(self.edStart1.GetValue()))
        self.UpdateDisp()

    def OnEdStep1KillFocus(self, event):
        self.scope.tpa.SetStepSize(1,float(self.edStep1.GetValue()))
        self.UpdateDisp()

    def OnChPiezo1Choice(self, event):
        self.scope.tpa.SetScanChannel(1,self.chPiezo1.GetSelection())
        self.UpdateDisp()

    def OnEdPhaseStep1KillFocus(self, event):
        event.Skip()

    def OnEdLambda1KillFocus(self, event):
        event.Skip()

    def OnEdPhaseMult1KillFocus(self, event):
        event.Skip()
