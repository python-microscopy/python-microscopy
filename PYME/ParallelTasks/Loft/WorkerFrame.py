#!/usr/bin/python

##################
# WorkerFrame.py
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

#Boa:Frame:WorkerMonFrame

import wx
import threading
import Pyro.core
import sys
import time
sys.path.append('.')
import matplotlib
matplotlib.use('WXAgg')
matplotlib.interactive(False)

#class workerClass(threading.Thread):
#    def run(self):
        
        
        #global guiThread
        #guiThread=wx

#        while 1:
            


def create(parent):
    return WorkerMonFrame(parent)

[wxID_WORKERMONFRAME, wxID_WORKERMONFRAMEBEXIT, 
] = [wx.NewId() for _init_ctrls in range(2)]

[wxID_WORKERMONFRAMETGETTASK] = [wx.NewId() for _init_utils in range(1)]

class WorkerMonFrame(wx.Frame):
    def _init_utils(self):
        # generated method, don't edit
        self.tGetTask = wx.Timer(id=wxID_WORKERMONFRAMETGETTASK, owner=self)
        self.Bind(wx.EVT_TIMER, self.OnTGetTaskTimer,
              id=wxID_WORKERMONFRAMETGETTASK)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_WORKERMONFRAME, name='WorkerMonFrame',
              parent=prnt, pos=wx.Point(430, 329), size=wx.Size(188, 139),
              style=wx.DEFAULT_FRAME_STYLE, title='Task Worker')
        self._init_utils()
        self.SetClientSize(wx.Size(180, 112))

        self.bExit = wx.Button(id=wxID_WORKERMONFRAMEBEXIT, label='Exit',
              name='bExit', parent=self, pos=wx.Point(0, 0), size=wx.Size(180,
              112), style=0)

    def __init__(self, parent):
        self._init_ctrls(parent)
        
        #wc = workerClass()
        #wc.start()
        Pyro.config.PYRO_MOBILE_CODE=1
        self.tq = Pyro.core.getProxyForURI("PYRONAME://taskQueue")
        self.tGetTask.Start(1000)

    def OnTGetTaskTimer(self, event):
        print 'Getting Task ...'
        time.sleep(0.01) #to improve GUI responsiveness
        self.tq.returnCompletedTask(self.tq.getTask()(gui=True))
        print 'Completed Task'
