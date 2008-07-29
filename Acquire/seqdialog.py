#Boa:Dialog:seqDialog

import wx
import simplesequenceaquisator

#redefine wxFrame with a version that hides when someone tries to close it
#dirty trick, but lets the Boa gui builder still work with frames we do this to
#NB must come after 'from wx.... import *' !!!
from noclosefr import * 

def create(parent):
    return seqDialog(parent)

[wxID_SEQDIALOG, wxID_SEQDIALOGBENDHERE, wxID_SEQDIALOGBMID_NUM, 
 wxID_SEQDIALOGBSTART, wxID_SEQDIALOGBSTARTHERE, wxID_SEQDIALOGBST_END, 
 wxID_SEQDIALOGCHPIEZO, wxID_SEQDIALOGSTATICBOX1, wxID_SEQDIALOGSTATICBOX2, 
 wxID_SEQDIALOGSTATICBOX3, wxID_SEQDIALOGSTATICBOX4, wxID_SEQDIALOGSTATICBOX5, 
 wxID_SEQDIALOGSTATICBOX6, wxID_SEQDIALOGSTMEMORY, wxID_SEQDIALOGTENDPOS, 
 wxID_SEQDIALOGTNUMSLICES, wxID_SEQDIALOGTSTEPSIZE, wxID_SEQDIALOGTSTPOS, 
] = map(lambda _init_ctrls: wx.NewId(), range(18))

class seqDialog(wxFrame):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_SEQDIALOG, name='seqDialog', parent=prnt,
              pos=wx.Point(374, 265), size=wx.Size(356, 194),
              style=wx.DEFAULT_FRAME_STYLE, title='Sequence')
        self.SetClientSize(wx.Size(348, 167))
        self.SetBackgroundColour(wx.Colour(209, 208, 203))

        self.chPiezo = wx.Choice(choices=[], id=wxID_SEQDIALOGCHPIEZO,
              name='chPiezo', parent=self, pos=wx.Point(136, 16), size=wx.Size(85,
              21), style=0)
        wx.EVT_CHOICE(self.chPiezo, wxID_SEQDIALOGCHPIEZO, self.OnChPiezoChoice)

        self.staticBox1 = wx.StaticBox(id=wxID_SEQDIALOGSTATICBOX1, label='Type',
              name='staticBox1', parent=self, pos=wx.Point(8, 0),
              size=wx.Size(112, 56), style=0)

        self.bSt_end = wx.RadioButton(id=wxID_SEQDIALOGBST_END,
              label='Start and End', name='bSt_end', parent=self,
              pos=wx.Point(16, 16), size=wx.Size(88, 13), style=0)
        self.bSt_end.SetValue(True)
        wx.EVT_RADIOBUTTON(self.bSt_end, wxID_SEQDIALOGBST_END,
              self.OnBSt_endRadiobutton)

        self.bMid_num = wx.RadioButton(id=wxID_SEQDIALOGBMID_NUM,
              label='Middle and #', name='bMid_num', parent=self,
              pos=wx.Point(16, 32), size=wx.Size(79, 13), style=0)
        self.bMid_num.SetValue(False)
        wx.EVT_RADIOBUTTON(self.bMid_num, wxID_SEQDIALOGBMID_NUM,
              self.OnBMid_numRadiobutton)

        self.staticBox2 = wx.StaticBox(id=wxID_SEQDIALOGSTATICBOX2,
              label='Piezo Channel', name='staticBox2', parent=self,
              pos=wx.Point(128, 0), size=wx.Size(112, 56), style=0)

        self.staticBox3 = wx.StaticBox(id=wxID_SEQDIALOGSTATICBOX3,
              label='Stop Pos (um)', name='staticBox3', parent=self,
              pos=wx.Point(128, 56), size=wx.Size(112, 48), style=0)

        self.tEndPos = wx.TextCtrl(id=wxID_SEQDIALOGTENDPOS, name='tEndPos',
              parent=self, pos=wx.Point(136, 72), size=wx.Size(56, 21), style=0,
              value='60')
        wx.EVT_KILL_FOCUS(self.tEndPos, self.OnTEndPosKillFocus)

        self.bEndHere = wx.Button(id=wxID_SEQDIALOGBENDHERE, label='Here',
              name='bEndHere', parent=self, pos=wx.Point(200, 72),
              size=wx.Size(32, 23), style=0)
        wx.EVT_BUTTON(self.bEndHere, wxID_SEQDIALOGBENDHERE, self.OnBEndHereButton)

        self.staticBox4 = wx.StaticBox(id=wxID_SEQDIALOGSTATICBOX4,
              label='Start Pos (um)', name='staticBox4', parent=self,
              pos=wx.Point(8, 56), size=wx.Size(112, 48), style=0)

        self.bStartHere = wx.Button(id=wxID_SEQDIALOGBSTARTHERE, label='Here',
              name='bStartHere', parent=self, pos=wx.Point(80, 72),
              size=wx.Size(32, 23), style=0)
        wx.EVT_BUTTON(self.bStartHere, wxID_SEQDIALOGBSTARTHERE,
              self.OnBStartHereButton)

        self.tStPos = wx.TextCtrl(id=wxID_SEQDIALOGTSTPOS, name='tStPos',
              parent=self, pos=wx.Point(16, 72), size=wx.Size(56, 21), style=0,
              value='40')
        wx.EVT_KILL_FOCUS(self.tStPos, self.OnTStPosKillFocus)

        self.staticBox5 = wx.StaticBox(id=wxID_SEQDIALOGSTATICBOX5,
              label='# Slices', name='staticBox5', parent=self, pos=wx.Point(248,
              56), size=wx.Size(88, 48), style=0)

        self.tNumSlices = wx.TextCtrl(id=wxID_SEQDIALOGTNUMSLICES,
              name='tNumSlices', parent=self, pos=wx.Point(256, 72),
              size=wx.Size(64, 24), style=0, value='100')
        self.tNumSlices.Enable(False)
        wx.EVT_KILL_FOCUS(self.tNumSlices, self.OnTNumSlicesKillFocus)

        self.staticBox6 = wx.StaticBox(id=wxID_SEQDIALOGSTATICBOX6,
              label=' Stepsize (um)', name='staticBox6', parent=self,
              pos=wx.Point(248, 0), size=wx.Size(88, 56), style=0)

        self.tStepSize = wx.TextCtrl(id=wxID_SEQDIALOGTSTEPSIZE,
              name='tStepSize', parent=self, pos=wx.Point(256, 16),
              size=wx.Size(64, 24), style=0, value='0.04')
        wx.EVT_KILL_FOCUS(self.tStepSize, self.OnTStepSizeKillFocus)

        self.bStart = wx.Button(id=wxID_SEQDIALOGBSTART, label='Go, Go, Go !!! ',
              name='bStart', parent=self, pos=wx.Point(128, 136), size=wx.Size(75,
              23), style=0)
        wx.EVT_BUTTON(self.bStart, wxID_SEQDIALOGBSTART, self.OnBStartButton)

        self.stMemory = wx.StaticText(id=wxID_SEQDIALOGSTMEMORY,
              label='staticText1', name='stMemory', parent=self, pos=wx.Point(48,
              112), size=wx.Size(264, 13), style=0)

    def __init__(self, parent, scope):
        self.scope = scope
        self._init_ctrls(parent)
        
        if not ('sa' in self.scope.__dict__):
            self.scope.sa = simplesequenceaquisator.SimpleSequenceAquisitor(self.scope.chaninfo, self.scope.cam, self.scope.piezos)
            
        for pz in self.scope.piezos:
            self.chPiezo.Append(pz[2])
        self.UpdateDisp()   
        
    def OnBEndHereButton(self, event):
        self.scope.sa.SetEndPos(self.scope.piezos[self.scope.sa.GetScanChannel()][0].GetPos(self.scope.piezos[self.scope.sa.GetScanChannel()][1]))
        self.UpdateDisp()
        #event.Skip()

    def OnBStartHereButton(self, event):
        self.scope.sa.SetStartPos(self.scope.piezos[self.scope.sa.GetScanChannel()][0].GetPos(self.scope.piezos[self.scope.sa.GetScanChannel()][1]))
        self.UpdateDisp()
        #event.Skip()

    def OnBStartButton(self, event):
        res = self.scope.sa.Verify()
        if res[0]:
            self.scope.pa.stop()

            #try:
            self.scope.sa.Prepare()
            #except:
            #    dialog = wxMessageDialog(None, 'The most likely reason is a lack of memory \nTry the following: Close any open aquisitions, Chose a ROI, Delete unnecessary channels, or decrease the # of slices', "Could not start aquisition", wx.OK)
            #    dialog.ShowModal()
            #    self.scope.sa.ds=[]
            #    self.scope.pa.Prepare(True)
            #    self.scope.pa.start()

            self.scope.sa.WantFrameNotification=[]
            self.scope.sa.WantFrameNotification.append(self.scope.aq_refr)

            self.scope.sa.WantStopNotification=[]
            self.scope.sa.WantStopNotification.append(self.scope.aq_end)

            self.scope.sa.start()

            self.scope.pb = wx.ProgressDialog('Aquisition in progress ...', 'Slice 1 of %d' % self.scope.sa.ds.getDepth(), self.scope.sa.ds.getDepth(), style = wx.PD_APP_MODAL|wx.PD_AUTO_HIDE|wx.PD_REMAINING_TIME|wx.PD_CAN_ABORT)
                
            
                
                
            
        else:
            dialog = wx.MessageDialog(None, res[2] + ' (%2.3f)'% res[3], "Parameter Error", wx.OK)
            dialog.ShowModal()
            
            if res[1] == 'StepSize':
                self.tStepSize.SetFocus()
            elif (self.scope.sa.GetStartMode() == self.scope.sa.CENTRE_AND_LENGTH):
                self.tNumSlices.SetFocus()
            elif (res[1] == 'StartPos'):
                self.tStPos.SetFocus()
            else:
                self.tEndPos.SetFocus() 
                 
        
        #event.Skip()

    def OnChPiezoChoice(self, event):
        self.scope.sa.SetScanChannel(self.chPiezo.GetSelection())
        self.UpdateDisp()
        #event.Skip()

    def OnBSt_endRadiobutton(self, event):
        self.scope.sa.SetStartMode(1)
        self.UpdateDisp()
        #event.Skip()

    def OnBMid_numRadiobutton(self, event):
        self.scope.sa.SetStartMode(0)
        self.UpdateDisp()
        #event.Skip()

    def OnTEndPosKillFocus(self, event):
        self.scope.sa.SetEndPos(float(self.tEndPos.GetValue()))
        self.UpdateDisp()
        #event.Skip()

    def OnTStPosKillFocus(self, event):
        self.scope.sa.SetStartPos(float(self.tStPos.GetValue()))
        self.UpdateDisp()
        #event.Skip()

    def OnTNumSlicesKillFocus(self, event):
        self.scope.sa.SetSeqLength(int(self.tNumSlices.GetValue()))
        self.UpdateDisp()
        #event.Skip()

    def OnTStepSizeKillFocus(self, event):
        self.scope.sa.SetStepSize(float(self.tStepSize.GetValue()))
        self.UpdateDisp()
        #event.Skip()
        
    def UpdateDisp(self):
        self.chPiezo.SetSelection(self.scope.sa.GetScanChannel())
            
        if self.scope.sa.GetStartMode() == self.scope.sa.START_AND_END:
            self.bSt_end.SetValue(True)
            self.bMid_num.SetValue(False)
            
            self.tNumSlices.Enable(False)
            self.tStPos.Enable(True)
            self.tEndPos.Enable(True)
            self.bStartHere.Enable(True)
            self.bEndHere.Enable(True)
        else:
            self.bSt_end.SetValue(False)
            self.bMid_num.SetValue(True)
            
            self.tNumSlices.Enable(True)
            self.tStPos.Enable(False)
            self.tEndPos.Enable(False)
            self.bStartHere.Enable(False)
            self.bEndHere.Enable(False)
        
        self.tStPos.SetValue('%2.3f' % self.scope.sa.GetStartPos())
        self.tEndPos.SetValue('%2.3f' % self.scope.sa.GetEndPos())
        self.tStepSize.SetValue('%2.3f' % self.scope.sa.GetStepSize())
        self.tNumSlices.SetValue('%d' % self.scope.sa.GetSeqLength())
        self.stMemory.SetLabel('Required Memory: %2.3f MB' % (self.scope.cam.GetPicWidth()*self.scope.cam.GetPicHeight()*self.scope.sa.GetSeqLength()*2*self.scope.sa.getReqMemChans(self.scope.sa.chans.cols)/(1024.0*1024.0)))

