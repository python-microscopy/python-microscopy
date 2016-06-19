#!/usr/bin/python

##################
# thorlabsPiezo.py
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

import win32com.client
from wx.lib import activexwrapper
import wx

PzWrapper = activexwrapper.MakeActiveXClass(
    win32com.client.gencache.GetClassForProgID('MGPIEZO.MGPiezoCtrl.1'),
    eventClass=None, eventObj=None)

SysWrapper = activexwrapper.MakeActiveXClass(
    win32com.client.gencache.GetClassForProgID('MG17SYSTEM.MG17SystemCtrl.1'),
    eventClass=None, eventObj=None)

def EnumeratePiezos():
    """Return serial numbers of all attached piezos"""

    f = wx.Frame(None)
    sy = SysWrapper(f)
    sy.StartCtrl()

    numPiezos = sy.GetNumHWUnits(win32com.client.constants.USB_PIEZO_DRIVE,1)[1]

    serialNums = []

    for i in range(numPiezos):
        serialNums.append(sy.GetHWSerialNum(win32com.client.constants.USB_PIEZO_DRIVE, i, 1)[1])

    f.Close()

    return serialNums
                          
class PiezoFrame(wx.Frame):
    def __init__(self, serialNumber, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.PiezoControl = PzWrapper(self, size=self.GetClientSize())

        self.PiezoControl.HWSerialNum = serialNumber
        self.PiezoControl.StartCtrl()
        #self.show()


class TLPiezo:
    def __init__(self, serialNumber, name=''):
        self.pzFrame = PiezoFrame(serialNumber, parent=None, size=(400,300), title=name)
        self.max_travel = self.pzFrame.PiezoControl.GetMaxTravel(0,1)[1]

        #set closed loop mode
        self.pzFrame.PiezoControl.SetControlMode(0,win32com.client.constants.CLOSED_LOOP)

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                self.pzFrame.PiezoControl.SetPosOutput(0,fPos)
            else:
                self.pzFrame.PiezoControl.SetPosOutput(0,self.max_travel)
        else:
            self.pzFrame.PiezoControl.SetPosOutput(0,0.0)

    def GetPos(self, iChannel=1):
        return self.pzFrame.PiezoControl.GetPosOutput(0,1)[1]

    def GetControlReady(self):
        return True

    def GetChannelObject(self):
        return 1

    def GetChannelPhase(self):
        return 1

    def GetMin(self,iChan=1):
        return 0

    def GetMax(self, iChan=1):
        return self.max_travel
