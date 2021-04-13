#!/usr/bin/python

##################
# stepDialog.py
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

#Boa:Dialog:stepDialog

#from wxPython.wx. import *
import wx
import math
import time

#redefine wx.Frame with a version that hides when someone tries to close it
#dirty trick, but lets the Boa gui builder still work with frames we do this to
#NB must come after 'from wx..... import *' !!!
#from noclosefr import * 

def create(parent):
    return stepDialog(parent)

[wxID_STEPDIALOG, wxID_STEPDIALOGBGOTO, wxID_STEPDIALOGBGOTOXY, 
 wxID_STEPDIALOGBON, wxID_STEPDIALOGCHSPEED, wxID_STEPDIALOGSTATICBOX1, 
 wxID_STEPDIALOGSTATICBOX2, wxID_STEPDIALOGSTATICBOX3, 
 wxID_STEPDIALOGSTATICBOX4, wxID_STEPDIALOGSTATICBOX5, wxID_STEPDIALOGTXPOS, 
 wxID_STEPDIALOGTYPOS, wxID_STEPDIALOGTZPOS, 
] = [wx.NewId() for i in range(13)]

class stepPanel(wx.Panel):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_STEPDIALOG,
              parent=prnt, size=wx.Size(183, 222))
        self.SetClientSize(wx.Size(175, 195))
        #self.SetBackgroundColour(wx.Colour(209, 208, 203))

        self.staticBox1 = wx.StaticBox(id=wxID_STEPDIALOGSTATICBOX1,
              label='YPos', name='staticBox1', parent=self, pos=wx.Point(8, 56),
              size=wx.Size(64, 48), style=0)

        self.tYPos = wx.TextCtrl(id=wxID_STEPDIALOGTYPOS, name='tYPos',
              parent=self, pos=wx.Point(16, 72), size=wx.Size(48, 24), style=0,
              value='0')
        self.tYPos.Enable(True)

        self.tZPos = wx.TextCtrl(id=wxID_STEPDIALOGTZPOS, name='tZPos',
              parent=self, pos=wx.Point(16, 120), size=wx.Size(48, 24), style=0,
              value='0')
        self.tZPos.Enable(True)

        self.staticBox2 = wx.StaticBox(id=wxID_STEPDIALOGSTATICBOX2,
              label='ZPos', name='staticBox2', parent=self, pos=wx.Point(8, 104),
              size=wx.Size(64, 48), style=0)

        self.tXPos = wx.TextCtrl(id=wxID_STEPDIALOGTXPOS, name='tXPos',
              parent=self, pos=wx.Point(16, 24), size=wx.Size(48, 24), style=0,
              value='0')
        self.tXPos.Enable(True)

        self.staticBox3 = wx.StaticBox(id=wxID_STEPDIALOGSTATICBOX3,
              label='XPos', name='staticBox3', parent=self, pos=wx.Point(8, 8),
              size=wx.Size(64, 48), style=0)

        self.chSpeed = wx.Choice(choices=['1', '2', '3', '4', '5', '6', '7', '8',
              '9', '10'], id=wxID_STEPDIALOGCHSPEED, name='chSpeed',
              parent=self, pos=wx.Point(96, 40), size=wx.Size(56, 21), style=0)
        self.chSpeed.SetSelection(0)
        self.chSpeed.Bind(wx.EVT_CHOICE, self.OnChoiceSpeed)

        self.staticBox4 = wx.StaticBox(id=wxID_STEPDIALOGSTATICBOX4,
              label='Speed', name='staticBox4', parent=self, pos=wx.Point(88,
              24), size=wx.Size(72, 48), style=0)

        self.bOn = wx.Button(id=wxID_STEPDIALOGBON, label='On', name='bOn',
              parent=self, pos=wx.Point(96, 80), size=wx.Size(56, 23), style=0)
        self.bOn.Bind(wx.EVT_BUTTON, self.OnButtonOn)

        self.staticBox5 = wx.StaticBox(id=wxID_STEPDIALOGSTATICBOX5,
              label='Joystick', name='staticBox5', parent=self, pos=wx.Point(80,
              8), size=wx.Size(88, 104), style=0)

        self.bGoto = wx.Button(id=wxID_STEPDIALOGBGOTO, label='Goto',
              name='bGoto', parent=self, pos=wx.Point(8, 160), size=wx.Size(64,
              23), style=0)
        self.bGoto.Bind(wx.EVT_BUTTON, self.OnBGotoButton)

        self.bGotoXY = wx.Button(id=wxID_STEPDIALOGBGOTOXY, label='Goto XY',
              name='bGotoXY', parent=self, pos=wx.Point(88, 120), size=wx.Size(75,
              23), style=0)
        self.bGotoXY.Bind(wx.EVT_BUTTON, self.OnBGotoXYButton)

    def __init__(self, parent, scope):
        self._init_ctrls(parent)
        self.scope = scope
        
        self.chSpeed.SetSelection(self.scope.step.GetJoystickSpeed() -1)
        
        if self.scope.step.GetJoystickStatus():
            self.bOn.SetLabel('Off')
        else:
            self.bOn.SetLabel('On')

    def OnChoiceSpeed(self, event):
        self.scope.step.SetJoystickSpeed(self.chSpeed.GetSelection() + 1)
        self.scope.step.SetMoveSpeed(self.chSpeed.GetSelection() + 1)
        #event.Skip()

    def OnButtonOn(self, event):
        self.scope.step.SetJoystickOnOff()
        if self.scope.step.GetJoystickStatus():
            self.bOn.SetLabel('Off')
        else:
            self.bOn.SetLabel('On')
        #event.Skip()

    def OnBGotoButton(self, event):
        # Goto -- with dodgy hack to get around problem with moves larger than 5mm
        #jstat = self.scope.step.GetJoystickStatus()
        #if (jstat):
        #    self.scope.step.SetJoystickOnOff()
        
        x0 = self.scope.step.GetPosX()
        y0 = self.scope.step.GetPosY()
        z0 = self.scope.step.GetPosZ()
        
        x = float(self.tXPos.GetValue())
        y = float(self.tYPos.GetValue())
        z = float(self.tZPos.GetValue())
        
        #print(('(%s, %s, %s)\n' % (x,y,z)))
        
        #dodgy hack ...
        dist = math.sqrt(math.pow(x - x0, 2) + math.pow(y - y0,2) + math.pow(z - z0,2))
        
        #while (dist > 4000):
        #    xt = x0 + (x - x0)*4000/dist
        #    yt = y0 + (y - y0)*4000/dist
        #    zt = z0 + (z - z0)*4000/dist
        #    
        #    self.scope.step.MoveTo(xt,yt,zt)
        #    
        #    time.sleep(1)
##            
##            self.scope.step.ContIO()
##            x0 = self.scope.step.GetPosX()
##            y0 = self.scope.step.GetPosY()
##            z0 = self.scope.step.GetPosZ()
##            
##            dist = math.sqrt(math.pow(x - x0, 2) + math.pow(y - y0,2) + math.pow(z - z0,2))
            
        self.scope.step.MoveTo(x,y,z)

        time.sleep(1)
            
        self.scope.step.ContIO()
        
        #if (jstat):
        #    self.scope.step.SetJoystickOnOff()
        #event.Skip()

    def OnBGotoXYButton(self, event):
        # Uses current ZPos
        #jstat = self.scope.step.GetJoystickStatus()
        #if (jstat):
        #    self.scope.step.SetJoystickOnOff()
            
        x0 = self.scope.step.GetPosX()
        y0 = self.scope.step.GetPosY()
        z0 = self.scope.step.GetPosZ()
        
        x = float(self.tXPos.GetValue())
        y = float(self.tYPos.GetValue())
        #z = float(self.tZPos.GetValue())
        z = self.scope.step.GetPosZ()
        
        #print(('(%s, %s, %s)\n' % (x,y,z)))
        
        #dodgy hack ...
        dist = math.sqrt(math.pow(x - x0, 2) + math.pow(y - y0,2) + math.pow(z - z0,2))
        #print(( 'dist = %s' % (dist)))
        
##        while (dist > 4000):
##            xt = x0 + (x - x0)*4000/dist
##            yt = y0 + (y - y0)*4000/dist
##            zt = z0 + (z - z0)*4000/dist
##            
##            print '(%s, %s, %s)\n' % (xt,yt,zt)
##            
##            self.scope.step.MoveTo(xt,yt,zt)
##            
##            time.sleep(1)
##            
##            self.scope.step.ContIO()
##            x0 = self.scope.step.GetPosX()
##            y0 = self.scope.step.GetPosY()
##            z0 = self.scope.step.GetPosZ()
##            
##            dist = math.sqrt(math.pow(x - x0, 2) + math.pow(y - y0,2) + math.pow(z - z0,2))
##            print 'dist = %s' % (dist)
            
        
        self.scope.step.MoveTo(x,y,z)

        time.sleep(1)
        self.scope.step.ContIO()
        
        #if (jstat):
        #    self.scope.step.SetJoystickOnOff()
        #event.Skip()
