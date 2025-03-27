#!/usr/bin/python

##################
# init_TIRF.py
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

from PYME.Acquire.Hardware.AndorIXon import AndorIXon
from PYME.Acquire.Hardware.AndorIXon import AndorControlFrame

from PYME.Acquire.Hardware import fakeShutters
import time

InitBG('EMCCD Camera', """
scope.cam = AndorIXon.iXonCamera()
""")
InitGUI("""
acf = AndorControlFrame.AndorPanel(MainFrame, scope.cam, scope)
camPanels.append((acf, 'Andor EMCCD Properties'))
""")

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [fakeShutters.CH1] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo
scope.shutters = fakeShutters




InitBG('Piezos', """
from PYME.Acquire.Hardware.Old import SMI1
scope.piez = SMI1.CPiezoOp()
scope.piez.Init(1)
scope.piez.SetMax(1, 100.)
scope.piez.SetMax(2, 100.)
scope.piezos.append((scope.piez, 2, 'Object'))
scope.piezos.append((scope.piez, 1, 'Y'))
#scope.piezos.append((scope.piez, 3, 'Phase'))
""")

InitGUI("""
from PYME.Acquire.Hardware import focusKeys
fk = focusKeys.FocusKeys(MainFrame, menuBar1, scope.piezos[0])
time1.register_callback(fk.refresh)
""")


#Stepper motor
InitBG('Stepper Motor', """
from PYME.Acquire.Hardware.Old import SMI1
import wx

scope.step = SMI1.CStepOp()
time1.register_callback(scope.step.ContIO)

mb = wx.MessageDialog(sh.GetParent(), 'Continue with Calibration of stage?\\nPLEASE CHECK that the slide holder has been removed\\n(and then press OK)', 'Stage Callibration', wx.YES_NO|wx.NO_DEFAULT)
ret = mb.ShowModal()
if (ret == wx.ID_YES):
    scope.step.Init(1)
else:
    scope.step.Init(2)

""")

InitBG('Lasers', """
from PYME.Acquire.Hardware import lasers

pport = lasers.PPort()
scope.l671 = lasers.ParallelSwitchedLaser('671',pport,0)
scope.l488 = lasers.ParallelSwitchedLaser('488',pport,1)

scope.lasers = [scope.l488,scope.l671]
""")

InitGUI("""
if 'lasers'in dir(scope):
    from PYME.Acquire.Hardware import LaserControlFrame
    lcf = LaserControlFrame.LaserControlLight(MainFrame,scope.lasers)
    time1.register_callback(lcf.refresh)
    toolPanels.append((lcf, 'Laser Control'))
""")






#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread
time.sleep(1)
scope.initDone = True
