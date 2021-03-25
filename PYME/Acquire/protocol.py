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
import logging
logger = logging.getLogger(__name__)
from PYME import config

from sys import maxsize as maxint

#minimal protocol which does nothing
class Protocol:
    def __init__(self, filename=None):
        # NOTE: .filename attribute is currently set in the spool controller, and will over-ride the filename passed to the constructor.
        # The filename parameter exists to allow setting the filename in protocols which are not instantiated through the spool controller, and
        # requires passing __name__ to the constructor in the protocol itself. Both solutions are a bit gross, and may be revisited in the future.
        self.filename = filename

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
    def __init__(self, taskList, metadataEntries = [], preflightList=[], 
                 filename=None):
        self.taskList = taskList
        Protocol.__init__(self, filename)
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
            eventLog.logEvent('ProtocolTask', '%d, %s, ' % (frameNum, t.what.__name__) + ', '.join([str(p) for p in t.params]))
            self.listPos += 1

    def OnFinish(self):
        while not  self.listPos >= len(self.taskList):
            t = self.taskList[self.listPos]
            self.listPos += 1
            t.what(*t.params)
            eventLog.logEvent('ProtocolTask', '%s, ' % ( t.what.__name__,) + ', '.join([str(p) for p in t.params]))
            





class ZStackTaskListProtocol(TaskListProtocol):
    def __init__(self, taskList, startFrame, dwellTime, metadataEntries=[], preflightList=[], randomise=False,
                 slice_order='saw', require_camera_restart=True, filename=None):
        """

        Parameters
        ----------
        taskList: list
            List of tasks
        startFrame: int
            Frame to begin acquisition
        dwellTime: int
            Number of frames to spent on each step
        metadataEntries: list
            List of metadata entries to propagate through spooling
        preflightList: list
            List of tasks to perform before starting, i.e. before frame zero
        randomise: bool
            Deprecated, use slice_order='random' instead
        slice_order: str
            Setting for reordering z steps.
                'saw': standard z-step moving linearly from one extreme to the other
                'random': acquire z steps in a random order
                'triangle': acquire z-steps in an up-then-down or down-then-up fashion like a triangle waveform
        require_camera_restart: bool
            Flag to toggle restarting the camera/frameWrangler on each step (True) or leave the camera running (False)
        """
        
        # add a check to ensure that dwell times are sensible
        preflightList.append(C('(self.dwellTime*scope.cam.GetIntegTime() > .1) or not scope.cam.contMode',
                               'Z step dwell time too short - increase either dwell time or integration time, or set camera mode to single shot / software triggered'))
        
        TaskListProtocol.__init__(self, taskList, metadataEntries, preflightList,
                                  filename)
        
        self.startFrame = startFrame
        self.dwellTime = dwellTime
        self.randomise = randomise
        if self.randomise:
            logger.warning("Use slice_order='random' instead of randomise=True")
            self.slice_order = 'random'
        else:
            self.slice_order = slice_order

        self.require_camera_restart = require_camera_restart

    def Init(self, spooler):
        stack_settings = getattr(spooler, 'stack_settings', scope.stackSettings)
        
        self.zPoss = np.arange(stack_settings.GetStartPos(),
                               stack_settings.GetEndPos() + .95 * stack_settings.GetStepSize(),
                               stack_settings.GetStepSize() * stack_settings.GetDirection())

        if self.slice_order != 'saw':
            if self.slice_order == 'random':
                self.zPoss = self.zPoss[np.argsort(np.random.rand(len(self.zPoss)))]
            elif self.slice_order == 'triangle':
                if len(self.zPoss) % 2:
                    # odd
                    self.zPoss = np.concatenate([self.zPoss[::2], self.zPoss[-2::-2]])
                else:
                    # even
                    self.zPoss = np.concatenate([self.zPoss[::2], self.zPoss[-1::-2]])


        self.piezoName = 'Positioning.%s' % stack_settings.GetScanChannel()
        self.startPos = scope.state[self.piezoName + '_target'] #FIXME - _target positions shouldn't be part of scope state
        self.pos = 0

        spooler.md.setEntry('Protocol.PiezoStartPos', self.startPos)
        spooler.md.setEntry('Protocol.ZStack', True)
        
        # Starting move relocated to execute after other -1 tasks as workaround for HTSMS system (see issue 766)
        # TODO - revisit the move
        #scope.state.setItem(self.piezoName, self.zPoss[self.pos], stopCamera=self.require_camera_restart)
        #eventLog.logEvent('ProtocolFocus', '%d, %3.3f' % (0, self.zPoss[self.pos]))

        TaskListProtocol.Init(self,spooler)

    def OnFrame(self, frameNum):
        if frameNum > self.startFrame:
            fn = int(floor((frameNum - self.startFrame)/self.dwellTime) % len(self.zPoss))
            if not fn == self.pos:
                self.pos = fn
                #self.piezo.MoveTo(self.piezoChan, self.zPoss[self.pos])
                scope.state.setItem(self.piezoName, self.zPoss[self.pos], stopCamera=self.require_camera_restart)
                eventLog.logEvent('ProtocolFocus', '%d, %3.3f' % (frameNum, self.zPoss[self.pos]))
                
        TaskListProtocol.OnFrame(self, frameNum)
        
        if frameNum == -1:
            # Make move to initial position **after** all other -1 setup tasks have been performed (in super-class
            # OnFrame() call above).
            # This is currently required as a work-around on the HTSMS system which needs to unlock the focus
            # lock before changing focus (issue 766).
            scope.state.setItem(self.piezoName, self.zPoss[self.pos], stopCamera=self.require_camera_restart)
            eventLog.logEvent('ProtocolFocus', '%d, %3.3f' % (-1, self.zPoss[self.pos]))

    def OnFinish(self):
        #return piezo to start position
        #self.piezo.MoveTo(self.piezoChan, self.startPos)
        scope.state.setItem(self.piezoName, self.startPos, stopCamera=self.require_camera_restart)

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
