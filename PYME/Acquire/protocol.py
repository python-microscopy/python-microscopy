#!/usr/bin/python

##################
# protocol.py
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

from math import floor
import numpy as np
from PYME.Acquire import eventLog

from PYME import config

from sys import maxsize as maxint

#minimal protocol which does nothing
class Protocol:
    def __init__(self):
        pass

    def Init(self, spooler):
        pass
    
    def OnFrame(self, frameNum):
        pass

    def PreflightCheck(self):
        return []

    def OnFinish(self):
        pass



NullProtocol = Protocol()

class TaskListTask:
    def __init__(self, when, what, *withParams):
        self.when = when
        self.what = what
        self.params = withParams

T = TaskListTask #to save typing in protocols

class PreflightCheckItem:
    def __init__(self, check, message):
        self.check = check
        self.message = message

C = PreflightCheckItem #to save typing in protocols

def Ex(str):
    exec(str)

def SetEMGain(emGain):
    scope.frameWrangler.stop()
    scope.cam.SetEMGain(emGain)
    scope.frameWrangler.start()
    
def SetIntegTime(iTime):
    scope.frameWrangler.stop()
    scope.cam.SetIntegTime(iTime)
    scope.frameWrangler.start()

def SetCameraShutter(open):
    scope.frameWrangler.stop()
    scope.cam.SetShutter(open)
    scope.frameWrangler.start()

def SetContinuousMode(contMode):
    if not (scope.cam.contMode == contMode):
        if contMode:
            scope.frameWrangler.stop()
            scope.cam.SetAcquisitionMode(scope.cam.MODE_CONTINOUS)
            #self.bUpdateInt.Enable(False)
            scope.frameWrangler.start()
        else:
            scope.frameWrangler.stop()
            scope.cam.SetAcquisitionMode(scope.cam.MODE_SINGLE_SHOT)
            #self.bUpdateInt.Enable(False)
            scope.frameWrangler.start()


class TaskListProtocol(Protocol):
    def __init__(self, taskList, metadataEntries = [], preflightList=[]):
        self.taskList = taskList
        Protocol.__init__(self)
        self.listPos = 0

        self.metadataEntries = metadataEntries
        self.preflightList = preflightList

    def PreflightCheck(self):
        failedChecks = []
        for c in self.preflightList:
            if not eval(c.check):
                failedChecks.append(c)

        return failedChecks

    def Init(self, spooler):
        self.listPos = 0

        self.OnFrame(-1) #do everything which needs to be done before acquisition starts

        spooler.md.setEntry('Protocol.Filename', self.filename)
        for e in self.metadataEntries:
            spooler.md.setEntry(*e)

    def OnFrame(self, frameNum):
        while not self.listPos >= len(self.taskList) and frameNum >= self.taskList[self.listPos].when:
            t = self.taskList[self.listPos]
            t.what(*t.params)
            eventLog.logEvent('ProtocolTask', '%d, %s, ' % (frameNum, t.what.__name__) + ', '.join(['%s' % p for p in t.params]))
            self.listPos += 1
        #print scope.pst.piezo.driftQueue.qsize()
        #if not scope.pst.piezo.driftQueue.empty():
        #    driftvalue = scope.pst.piezo.driftQueue.get()
        #    eventLog.logEvent('ShiftMeasure', '%3.4f, %3.4f, %3.4f, %3.4f' % driftvalue)

    def OnFinish(self):
        while not  self.listPos >= len(self.taskList):
            t = self.taskList[self.listPos]
            self.listPos += 1
            t.what(*t.params)
            eventLog.logEvent('ProtocolTask', '%s, ' % ( t.what.__name__,) + ', '.join(['%s' % p for p in t.params]))
            





class ZStackTaskListProtocol(TaskListProtocol):
    def __init__(self, taskList, startFrame, dwellTime, metadataEntries = [], preflightList=[], randomise = False):
        TaskListProtocol.__init__(self, taskList, metadataEntries, preflightList)
        
        self.startFrame = startFrame
        self.dwellTime = dwellTime
        self.randomise = randomise

    def Init(self, spooler):
        #self.zPoss = np.arange(scope.sa.GetStartPos(), scope.sa.GetEndPos()+.95*scope.sa.GetStepSize(),scope.sa.GetStepSize())

      
        # zigzag stepping
        zPoss1 = np.arange(scope.stackSettings.GetStartPos(), scope.stackSettings.GetEndPos()+.95*scope.stackSettings.GetStepSize(),scope.stackSettings.GetStepSize()*scope.stackSettings.GetDirection())
        zPoss2 = np.arange(scope.stackSettings.GetEndPos(), scope.stackSettings.GetStartPos()-.95*scope.stackSettings.GetStepSize(),-1*scope.stackSettings.GetStepSize()*scope.stackSettings.GetDirection())
        #self.zPoss = np.asarray(zPoss1.tolist()+zPoss2.tolist()) # this should be doable more directly with array functions
        self.zPoss = np.append(zPoss1, zPoss2)

        if self.randomise:
            self.zPoss = self.zPoss[np.argsort(np.random.rand(len(self.zPoss)))]

        #piezo = scope.positioning[scope.stackSettings.GetScanChannel()]
        self.piezoName = 'Positioning.%s' % scope.stackSettings.GetScanChannel()
        #self.piezo = piezo[0]
        #self.piezoChan = piezo[1]
        #self.startPos = self.piezo.GetPos(self.piezoChan)
        self.startPos = scope.state[self.piezoName]
        self.pos = 0

        spooler.md.setEntry('Protocol.PiezoStartPos', self.startPos)
        spooler.md.setEntry('Protocol.ZStack', True)
        
        scope.state.setItem(self.piezoName, self.zPoss[self.pos], stopCamera=True)
        eventLog.logEvent('ProtocolFocus', '%d, %3.3f' % (0, self.zPoss[self.pos]))

        TaskListProtocol.Init(self,spooler)

    def OnFrame(self, frameNum):
        if frameNum > self.startFrame:
            fn = floor((frameNum - self.startFrame)/self.dwellTime) % len(self.zPoss)
            if not fn == self.pos:
                self.pos = fn
                #self.piezo.MoveTo(self.piezoChan, self.zPoss[self.pos])
                scope.state.setItem(self.piezoName, self.zPoss[self.pos], stopCamera=True)
                eventLog.logEvent('ProtocolFocus', '%d, %3.3f' % (frameNum, self.zPoss[self.pos]))
                
        TaskListProtocol.OnFrame(self, frameNum)

    def OnFinish(self):
        #return piezo to start position
        #self.piezo.MoveTo(self.piezoChan, self.startPos)
        scope.state.setItem(self.piezoName, self.startPos, stopCamera=True)

        TaskListProtocol.OnFinish(self)

NullZProtocol = ZStackTaskListProtocol([], 0, 100)
NullZProtocol.filename = '<none>'


def _get_protocol_dict():
    import glob, os
    prot_dir = (os.path.dirname(__file__)) + '/Protocols/[a-zA-Z]*.py'
    #print prot_dir
    protocolList = glob.glob(prot_dir)
    protocolDict = {os.path.split(p)[-1] : p for p in protocolList}

    # in this implementation custom protocols overwrite protocols of the same name
    import PYME.config
    customPDict = PYME.config.get_custom_protocols()
    for p in customPDict.keys():
        protocolDict[p] = customPDict[p]

    return protocolDict

def get_protocol_list():
    protocolList = ['<None>', ] + sorted(list(_get_protocol_dict().keys()))

    return protocolList

def get_protocol(protocol_name, reloadProtocol=True):
    import imp

    pmod = imp.load_source(protocol_name.split('.')[0], _get_protocol_dict()[protocol_name])

    #if reloadProtocol:
    #    reload(pmod)

    return pmod
