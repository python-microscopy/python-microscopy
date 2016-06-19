#!/usr/bin/python

##################
# timeseqdialog.py
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

#Boa:Dialog:seqDialog
#from wxPython.wx import *
import wx
import timesequenceaquisator
#redefine wxFrame with a version that hides when someone tries to close it
#dirty trick, but lets the Boa gui builder still work with frames we do this to
#NB must come after 'from wx.... import *' !!!
from noclosefr import * 
def create(parent):
    return seqDialog(parent)
[wxID_TSEQDIALOG, wxID_TSEQDIALOGBSTART, wxID_TSEQDIALOGSTATICBOX5, 
 wxID_TSEQDIALOGSTATICBOX6, wxID_TSEQDIALOGSTMEMORY, wxID_TSEQDIALOGTNUMSLICES, 
 wxID_TSEQDIALOGTTIMESTEP, 
] = map(lambda _init_ctrls: wx.NewId(), range(7))
class seqDialog(wxFrame):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wxFrame.__init__(self, id=wxID_TSEQDIALOG, name='tseqDialog', parent=prnt,
              pos=wx.Point(374, 265), size=wx.Size(259, 160),
              style=wx.DEFAULT_FRAME_STYLE, title='Time Sequence')
        self.SetClientSize(wx.Size(251, 133))
        self.SetBackgroundColour(wx.Colour(209, 208, 203))
        self.staticBox5 = wx.StaticBox(id=wxID_TSEQDIALOGSTATICBOX5,
              label='# Slices', name='staticBox5', parent=self, pos=wx.Point(24,
              8), size=wx.Size(88, 48), style=0)
        self.tNumSlices = wx.TextCtrl(id=wxID_TSEQDIALOGTNUMSLICES,
              name='tNumSlices', parent=self, pos=wx.Point(40, 24),
              size=wx.Size(64, 24), style=0, value='100')
        #self.tNumSlices.Enable(False)
        wx.EVT_KILL_FOCUS(self.tNumSlices, self.OnTNumSlicesKillFocus)
        self.staticBox6 = wx.StaticBox(id=wxID_TSEQDIALOGSTATICBOX6,
              label=' Time step (s)', name='staticBox6', parent=self,
              pos=wx.Point(128, 8), size=wx.Size(88, 48), style=0)
        self.tTimeStep = wx.TextCtrl(id=wxID_TSEQDIALOGTTIMESTEP,
              name='tTimeStep', parent=self, pos=wx.Point(136, 24),
              size=wx.Size(64, 24), style=0, value='0.1')
        wx.EVT_KILL_FOCUS(self.tTimeStep, self.OnTStepSizeKillFocus)
        self.bStart = wx.Button(id=wxID_TSEQDIALOGBSTART, label='Go, Go, Go !!! ',
              name='bStart', parent=self, pos=wx.Point(80, 104), size=wx.Size(75,
              23), style=0)
        wx.EVT_BUTTON(self.bStart, wxID_TSEQDIALOGBSTART, self.OnBStartButton)
        self.stMemory = wx.StaticText(id=wxID_TSEQDIALOGSTMEMORY,
              label='staticText1', name='stMemory', parent=self, pos=wx.Point(8,
              72), size=wx.Size(232, 16), style=0)
    def __init__(self, parent, scope):
        self.scope = scope
        self._init_ctrls(parent)
        
        if not ('ta' in self.scope.__dict__):
            self.scope.ta = timesequenceaquisator.TimeSequenceAquisitor(self.scope.chaninfo, self.scope.cam, self.scope.shutters)
            
        self.UpdateDisp()   
        
    
    def OnBStartButton(self, event):
        res = self.scope.ta.Verify()
        if res[0]:
            self.scope.frameWrangler.stop()
            #try:
            self.scope.ta.Prepare()
            #except:
            #    dialog = wxMessageDialog(None, 'The most likely reason is a lack of memory \nTry the following: Close any open aquisitions, Chose a ROI, Delete unnecessary channels, or decrease the # of slices', "Could not start aquisition", wx.OK)
            #    dialog.ShowModal()
            #    self.scope.ta.ds=[]
            #    self.scope.frameWrangler.Prepare(True)
            #    self.scope.frameWrangler.start()
            self.scope.ta.WantFrameNotification=[]
            self.scope.ta.WantFrameNotification.append(self.scope.aqt_refr)
            self.scope.ta.WantStopNotification=[]
            self.scope.ta.WantStopNotification.append(self.scope.aqt_end)
            self.scope.ta.start()
            self.scope.pb = wx.ProgressDialog('Aquisition in progress ...', 'Slice 1 of %d' % self.scope.ta.ds.getDepth(), self.scope.ta.ds.getDepth(), style = wx.PD_APP_MODAL|wx.PD_AUTO_HIDE|wx.PD_REMAINING_TIME|wx.PD_CAN_ABORT)
                
            
                
                
            
        else:
            dialog = wx.MessageDialog(None, res[2] + ' (%2.3f)'% res[3], "Parameter Error", wx.OK)
            dialog.ShowModal()
            
            if res[1] == 'StepSize':
                self.tStepSize.SetFocus()
            elif (self.scope.ta.GetStartMode() == self.scope.ta.CENTRE_AND_LENGTH):
                self.tNumSlices.SetFocus()
            elif (res[1] == 'StartPos'):
                self.tStPos.SetFocus()
            else:
                self.tEndPos.SetFocus() 
                 
        
        #event.Skip()
    
    def OnTNumSlicesKillFocus(self, event):
        self.scope.ta.SetSeqLength(int(self.tNumSlices.GetValue()))
        self.UpdateDisp()
        #event.Skip()
    def OnTStepSizeKillFocus(self, event):
        self.scope.ta.SetTimeStep(float(self.tTimeStep.GetValue()))
        self.UpdateDisp()
        #event.Skip()
        
    def UpdateDisp(self):
            
        self.tTimeStep.SetValue('%2.3f' % self.scope.ta.GetTimeStep())
        self.tNumSlices.SetValue('%d' % self.scope.ta.GetSeqLength())
        self.stMemory.SetLabel('Required Memory: %2.3f MB' % (self.scope.cam.GetPicWidth()*self.scope.cam.GetPicHeight()*self.scope.ta.GetSeqLength()*2*self.scope.ta.getReqMemChans(self.scope.ta.chans.cols)/(1024.0*1024.0)))
