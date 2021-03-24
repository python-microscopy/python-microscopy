#!/usr/bin/python

##################
# simplesequenceaquisator.py
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

import math

import time
from PYME.IO import MetaDataHandler
from PYME.util import webframework

import logging
logger = logging.getLogger(__name__)

import threading

class StackSettings(object):
    """A class to keep settings for acquiring z-stacks"""
    # 'Constants'
    PHASE = 1
    OBJECT = 0
    CENTRE_AND_LENGTH = 0
    START_AND_END = 1
    FORWARDS = 1
    BACKWARDS = -1
    
    SCAN_MODES = ['Middle and Number', 'Start and End']
    
    DEFAULTS = {
        'StartPos': 0,
        'EndPos': 0,
        'StepSize': 0.2,
        'NumSlices': 100,
        'ScanMode': SCAN_MODES[0],
        'ScanPiezo': 'z',
        'DwellFrames': -1,
    }
        
    def __init__(self,scope, **kwargs):
        """
        Create a stack settings object.
        
        NB - extra args map 1:1 to stack metadata entries. Start and end pos are ignored if ScanMode = 'Middle and Number'
        
        Parameters
        ----------
        scope
        ScanMode
        StartPos
        EndPos
        StepSize
        NumSlices
        ScanPiezo
        """
        #PreviewAquisator.__init__(self, chans, cam, shutters, None)
        self.scope= scope 
        #self.log = _log
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.ProvideStackMetadata)

        self._settings_changed = threading.Condition()
        
        d1 = dict(self.DEFAULTS)
        d1.update(kwargs)
        
        self.update(**d1)
        
        self.direction = self.FORWARDS
        
        from PYME.Acquire import webui
        # add webui endpoints (if running under webui)
        webui.add_endpoints(self, '/stack_settings')

    
    def update(self, ScanMode=None, StartPos=None, EndPos=None, StepSize=None, NumSlices=None, ScanPiezo=None, DwellFrames=None):
        if ScanPiezo is not None:
            self.ScanChan = ScanPiezo
        if ScanMode is not None:
            self.StartMode = self.SCAN_MODES.index(ScanMode)
        if NumSlices is not None:
            self.SeqLength = int(NumSlices)
        if StepSize is not None:
            self.StepSize = float(StepSize)
        if StartPos is not None:
            self.startPos = float(StartPos)
        if EndPos is not None:
            self.endPos = float(EndPos)
        if DwellFrames is not None:
            self._dwell_frames = float(DwellFrames)
            
        with self._settings_changed:
            self._settings_changed.notify_all()

    @webframework.register_endpoint('/update', output_is_json=False)
    def update_json(self, body):
        import json
        self.update(**json.loads(body))

    @webframework.register_endpoint('/settings', output_is_json=False)
    def settings(self):
        return {
            'StartPos' : self.GetStartPos(),
            'EndPos' : self.GetEndPos(),
            'StepSize': self.GetStepSize(),
            'NumSlices' : self.GetSeqLength(),
            'ScanMode': self.SCAN_MODES[self.GetStartMode()],
            'ScanPiezo' : self.GetScanChannel(),
            'DwellFrames' : self._dwell_frames,
        }

    @webframework.register_endpoint('/settings_longpoll', output_is_json=False)
    def settings_longpoll(self):
        with self._settings_changed:
            self._settings_changed.wait()
        
        return self.settings()
        
                         
    def GetScanChannel(self):
        return self.ScanChan
    
    def piezoGoHome(self):
        #self.piezos[self.GetScanChannel()][0].MoveTo(self.piezos[self.GetScanChannel()][1],self.GetPrevPos(),False)
        self.scope.SetPos(**{self.GetScanChannel() : self.GetPrevPos()})
    
    def SetScanChannel(self,iMode):
        self.ScanChan = iMode
        with self._settings_changed:
            self._settings_changed.notify_all()
    
    def SetSeqLength(self,iLength):
        self.SeqLength = int(iLength)
        with self._settings_changed:
            self._settings_changed.notify_all()
        
    def GetSeqLength(self):
        if (self.StartMode == 0):
            return self.SeqLength
        else:
            return int(math.ceil(abs(self.GetEndPos() - self.GetStartPos())/self.GetStepSize()))
            
    def SetStartMode(self, iMode):
        self.StartMode = iMode
        with self._settings_changed:
            self._settings_changed.notify_all()
        
    def GetStartMode(self):
        return self.StartMode
    
    def SetStepSize(self, fSize):
        self.StepSize = float(fSize)
        with self._settings_changed:
            self._settings_changed.notify_all()
    
    def GetStepSize(self):
        return self.StepSize
    
    def SetStartPos(self, sPos):
        self.startPos = float(sPos)
        with self._settings_changed:
            self._settings_changed.notify_all()
        
    def GetStartPos(self):
        if (self.GetStartMode() == 0):
            return self._CurPos() - (self.GetStepSize()*(self.GetSeqLength() - 1)*self.GetDirection()/2)
        else: 
            if not ("startPos" in dir(self)):
                raise RuntimeError("Please call SetStartPos first !!")
            return self.startPos
            
    def SetEndPos(self, ePos):
        self.endPos = float(ePos)
        with self._settings_changed:
            self._settings_changed.notify_all()
        
    def GetEndPos(self):
        if (self.GetStartMode() == 0):
            return self._CurPos() + (self.GetStepSize()*(self.GetSeqLength() - 1)*self.GetDirection()/2)
        else: 
            if not ("endPos" in dir(self)):
                raise RuntimeError("Please call SetEndPos first !!")
            return self.endPos
        
    def SetPrevPos(self, sPos):
        self.prevPos = sPos
        
    def GetPrevPos(self):
        if not ("prevPos" in dir(self)):
            raise RuntimeError("Please call SetPrevPos first !!")
        return self.prevPos
    
    def SetDirection(self, dir):
        " Fowards = 1, backwards = -1 "
        self.direction = dir
        with self._settings_changed:
            self._settings_changed.notify_all()
        
    def GetDirection(self):
        if (self.GetStartMode() == 0):
            if (self.direction > 0.1):
                return 1
            else:
                return -1
        else:
            if ((self.GetEndPos() - self.GetStartPos()) > 0.1):
                return 1
            else:
                return -1
            
    def _CurPos(self):
        #return self.piezos[self.GetScanChannel()][0].GetPos(self.piezos[self.GetScanChannel()][1])
        return self.scope.GetPos()[self.GetScanChannel()]
            
    def Verify(self):
        rmin, rmax = self.scope.GetPosRange()[self.GetScanChannel()]
        if (self.GetStartPos() < rmin):
            return (False, 'StartPos', 'StartPos is smaller than piezo minimum',rmin) 
        
        if (self.GetStartPos() > rmax):
            return (False, 'StartPos', 'StartPos is larger than piezo maximum',rmax) 
        if (self.GetEndPos() < rmin):
            return (False, 'EndPos', 'EndPos is smaller than piezo minimum',rmin) 
        
        if (self.GetEndPos() > rmax):
            return (False, 'EndPos', 'EndPos is larger than piezo maximum',rmax) 
        
        #if (self.GetEndPos() < self.GetStartPos()):
        #    return (False, 'EndPos', 'EndPos is before Startpos',self.GetStartPos()) 
        #stepsize limits are at present arbitrary == not really restricted to sensible values
        #if (self.GetStepSize() < 0.001):
        #    return (False, 'StepSize', 'StepSize is smaller than piezo minimum',0.001) 
        
        #if (self.GetStartPos() > 90):
        #    return (False, 'StepSize', 'Simplesequenceaquisator StepSize is larger than piezo maximum',90)
        
        return (True,) 

    
    def ProvideStackMetadata(self, mdh):
        try:
            mdh.setEntry('StackSettings.StartPos', self.GetStartPos())
            mdh.setEntry('StackSettings.EndPos', self.GetEndPos())
            mdh.setEntry('StackSettings.StepSize', self.GetStepSize())
            mdh.setEntry('StackSettings.NumSlices', self.GetSeqLength())
            mdh.setEntry('StackSettings.ScanMode', self.SCAN_MODES[self.GetStartMode()])
            mdh.setEntry('StackSettings.ScanPiezo', self.GetScanChannel())
    
            mdh.setEntry('voxelsize.z', self.GetStepSize())
        except:
            logger.exception('Error writing stack metadata')


    def doStartLog(self):
        #new metadata handling
        self.mdh.setEntry('StartTime', time.time())

        self.mdh.setEntry('AcquisitionType', 'Stack')

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(self.mdh)



    def doStopLog(self):
        self.mdh.setEntry('EndTime', time.time())

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.mdh)


   
