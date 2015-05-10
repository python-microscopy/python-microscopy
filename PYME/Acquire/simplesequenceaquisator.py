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
#import datetime
from PYME.Acquire.previewaquisator import PreviewAquisator
import time
from PYME.Acquire import MetaDataHandler

class SimpleSequenceAquisitor(PreviewAquisator):
    # 'Constants'
    PHASE = 1
    OBJECT = 0
    CENTRE_AND_LENGTH = 0
    START_AND_END = 1
    FORWARDS = 1
    BACKWARDS = -1
        
    def __init__(self,chans, cam, shutters,_piezos, _log={}):
        PreviewAquisator.__init__(self, chans, cam, shutters, None)
        self.piezos = _piezos
        self.log = _log
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.ProvideStackMetadata)
        
        self.ScanChan  = 0
        self.StartMode = self.CENTRE_AND_LENGTH
        self.SeqLength = 100  
        self.StepSize  = 0.2
        self.startPos = 0
        self.endPos = 0
        
        self.direction = self.FORWARDS
        
    def doPiezoStep(self):
        if(self.GetStepSize() > 0):
            #self.piezos[self.GetScanChannel()][0].MoveTo(self.piezos[self.GetScanChannel()][1], self.piezos[self.GetScanChannel()][0].GetPos(self.piezos[self.GetScanChannel()][1]) + 
            #    self.GetDirection()*self.GetStepSize(), False)
            self.piezos[self.GetScanChannel()][0].MoveTo(self.piezos[self.GetScanChannel()][1], self.startPosRef + self.GetDirection()*self.GetStepSize()*(self.ds.getZPos() +1), False)
            #time.sleep(0.01)
            #print self.piezos[self.GetScanChannel()][0].GetPos(self.piezos[self.GetScanChannel()][1])
            #print self.GetDirection()*self.GetStepSize()*(self.ds.getZPos() +1)
            #print self.startPosRef
            #print self.GetScanChannel()
    def piezoReady(self):
        return not ((self.GetStepSize() > 0) and
            (self.piezos[self.GetScanChannel()][0].GetControlReady() == False))
    def setPiezoStartPos(self):
        " Used internally to move the piezo at the beginning of the aquisition "
        #self.curpos = self.piezos[self.GetScanChannel()][0].GetPos(self.piezos[self.GetScanChannel()][1])
        self.SetPrevPos(self._CurPos()) # aktuelle Position der Piezotische merken
        self.startPosRef = self.GetStartPos()       
        self.piezos[self.GetScanChannel()][0].MoveTo(self.piezos[self.GetScanChannel()][1], self.GetStartPos(), False)
            
    def getNextDsSlice(self):
        return self.ds.nextZ()
    def GetScanChannel(self):
        return self.ScanChan
    def piezoGoHome(self):
        self.piezos[self.GetScanChannel()][0].MoveTo(self.piezos[self.GetScanChannel()][1],self.GetPrevPos(),False)
    def SetScanChannel(self,iMode):
        self.ScanChan = iMode
    def SetSeqLength(self,iLength):
        self.SeqLength = iLength
    def GetSeqLength(self):
        if (self.StartMode == 0):
            return self.SeqLength
        else:
            return int(math.ceil(abs(self.GetEndPos() - self.GetStartPos())/self.GetStepSize()))
    def SetStartMode(self, iMode):
        self.StartMode = iMode
    def GetStartMode(self):
        return self.StartMode
    def SetStepSize(self, fSize):
        self.StepSize = fSize
    def GetStepSize(self):
        return self.StepSize
    def SetStartPos(self, sPos):
        self.startPos = sPos
    def GetStartPos(self):
        if (self.GetStartMode() == 0):
            return self._CurPos() - (self.GetStepSize()*(self.GetSeqLength() - 1)*self.GetDirection()/2)
        else: 
            if not ("startPos" in dir(self)):
                raise RuntimeError("Please call SetStartPos first !!")
            return self.startPos
    def SetEndPos(self, ePos):
        self.endPos = ePos
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
        return self.piezos[self.GetScanChannel()][0].GetPos(self.piezos[self.GetScanChannel()][1])
            
    def Verify(self):
        if (self.GetStartPos() < self.piezos[self.GetScanChannel()][0].GetMin(self.piezos[self.GetScanChannel()][1])):
            return (False, 'StartPos', 'StartPos is smaller than piezo minimum',self.piezos[self.GetScanChannel()][0].GetMin(self.piezos[self.GetScanChannel()][1])) 
        
        if (self.GetStartPos() > self.piezos[self.GetScanChannel()][0].GetMax(self.piezos[self.GetScanChannel()][1])):
            return (False, 'StartPos', 'StartPos is larger than piezo maximum',self.piezos[self.GetScanChannel()][0].GetMax(self.piezos[self.GetScanChannel()][1])) 
        if (self.GetEndPos() < self.piezos[self.GetScanChannel()][0].GetMin(self.piezos[self.GetScanChannel()][1])):
            return (False, 'EndPos', 'EndPos is smaller than piezo minimum',self.piezos[self.GetScanChannel()][0].GetMin(self.piezos[self.GetScanChannel()][1])) 
        
        if (self.GetEndPos() > self.piezos[self.GetScanChannel()][0].GetMax(self.piezos[self.GetScanChannel()][1])):
            return (False, 'EndPos', 'EndPos is larger than piezo maximum',self.piezos[self.GetScanChannel()][0].GetMax(self.piezos[self.GetScanChannel()][1])) 
        
        #if (self.GetEndPos() < self.GetStartPos()):
        #    return (False, 'EndPos', 'EndPos is before Startpos',self.GetStartPos()) 
        #stepsize limits are at present arbitrary == not really restricted to sensible values
        #if (self.GetStepSize() < 0.001):
        #    return (False, 'StepSize', 'StepSize is smaller than piezo minimum',0.001) 
        
        #if (self.GetStartPos() > 90):
        #    return (False, 'StepSize', 'Simplesequenceaquisator StepSize is larger than piezo maximum',90)
        
        return (True,) 

    
    def ProvideStackMetadata(self, mdh):
        mdh.setEntry('StackSettings.StartPos', self.GetStartPos())
        mdh.setEntry('StackSettings.EndPos', self.GetEndPos())
        mdh.setEntry('StackSettings.StepSize', self.GetStepSize())
        mdh.setEntry('StackSettings.NumSlices', self.GetSeqLength())
        mdh.setEntry('StackSettings.ScanMode', ['Middle and Number', 'Start and End'][self.GetStartMode()])
        mdh.setEntry('StackSettings.ScanPiezo', self.piezos[self.GetScanChannel()][2])

        mdh.setEntry('voxelsize.z', self.GetStepSize())


    def doStartLog(self):
        #new metadata handling
        self.mdh.setEntry('StartTime', time.time())

        self.mdh.setEntry('AcquisitionType', 'Stack')

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStartMetadata:
            mdgen(self.mdh)

        ##############################
        #old log stuff follows (DEPRECATED!!!)
#        if not 'GENERAL' in self.log.keys():
#            self.log['GENERAL'] = {}
#        if not 'PIEZOS' in self.log.keys():
#            self.log['PIEZOS'] = {}
#        if not 'CAMERA' in self.log.keys():
#            self.log['CAMERA'] = {}
#
#        self.log['PIEZOS']['Piezo'] = self.piezos[self.GetScanChannel()][2]
#        self.log['PIEZOS']['Stepsize'] = self.GetStepSize()
#        self.log['PIEZOS']['StartPos'] = self.GetStartPos()
#        for pz in self.piezos:
#            self.log['PIEZOS']['%s_Pos' % pz[2]] = pz[0].GetPos(pz[1])
#
#        #self.log['STEPPER']['XPos'] = m_pDoc->Step.GetNullX(),
#        #m_pDoc->Step.GetPosX(), m_pDoc->Step.GetNullY(), m_pDoc->Step.GetPosY(),
#        #m_pDoc->Step.GetNullZ(), m_pDoc->Step.GetPosZ());
#        self.cam.GetStatus()
#        self.log['CAMERA']['Binning'] = self.cam.GetHorizBin()
#        #m_pDoc->LogData.SetIntegTime(m_pDoc->Camera.GetIntegTime());
#        self.log['GENERAL']['Width'] = self.cam.GetPicWidth()
#        self.log['GENERAL']['Height'] = self.cam.GetPicHeight()
#        self.log['CAMERA']['ROIPosX'] = self.cam.GetROIX1()
#        self.log['CAMERA']['ROIPosY'] = self.cam.GetROIY1()
#        self.log['CAMERA']['ROIWidth'] = self.cam.GetROIX2() - self.cam.GetROIX1()
#        self.log['CAMERA']['ROIHeight'] = self.cam.GetROIY2() - self.cam.GetROIY1()
#        self.log['CAMERA']['StartCCDTemp'] = self.cam.GetCCDTemp()
#        self.log['CAMERA']['StartElectrTemp'] = self.cam.GetElectrTemp()
#
#        self.log['GENERAL']['NumChannels'] = self.ds.getNumChannels()
#        self.log['GENERAL']['NumHWChans'] = self.numHWChans
#
#        for ind in range(self.numHWChans):
#            self.log['SHUTTER_%d' % ind] = {}
#            self.log['SHUTTER_%d' % ind]['Name'] = self.chans.names[ind]
#            self.log['SHUTTER_%d' % ind]['IntegrationTime'] = self.chans.itimes[ind]
#            self.log['SHUTTER_%d' % ind]['Mask'] = self.hwChans[ind]
#
#            s = ''
#            bef = 0
#            if (self.cols[ind] & self.BW):
#                s = s + 'BW'
#                bef = 1
#            if (self.cols[ind] & self.RED):
#                if bef:
#                    s = s + ' '
#                s = s + 'R'
#                bef = 1
#            if (self.cols[ind] & self.GREEN1):
#                if bef:
#                    s = s + ' '
#                s = s + 'G1'
#                bef = 1
#            if (self.cols[ind] & self.GREEN2):
#                if bef:
#                    s = s + ' '
#                s = s + 'G2'
#                bef = 1
#            if (self.cols[ind] & self.BLUE):
#                if bef:
#                    s = s + ' '
#                s = s + 'B'
#                #bef = 1
#            self.log['SHUTTER_%d' % ind]['Colours'] = s
#
#        dt = datetime.datetime.now()
#
#        self.log['GENERAL']['Date'] = '%d/%d/%d' % (dt.day, dt.month, dt.year)
#        self.log['GENERAL']['StartTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)
        #m_pDoc->LogData.SaveSeqROIMode(m_pDoc->Camera.GetROIMode());
        #pass

    def doStopLog(self):
        self.mdh.setEntry('EndTime', time.time())

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.mdh)

#        self.log['GENERAL']['Depth'] = self.ds.getDepth()
#        self.log['PIEZOS']['EndPos'] = self.GetEndPos()
#        self.cam.GetStatus()
#        self.log['CAMERA']['EndCCDTemp'] = self.cam.GetCCDTemp()
#        self.log['CAMERA']['EndElectrTemp'] = self.cam.GetElectrTemp()
#
#        dt = datetime.datetime.now()
#        self.log['GENERAL']['EndTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)
  
   
