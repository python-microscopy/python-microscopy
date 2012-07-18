#!/usr/bin/python

##################
# timesequenceaquisator.py
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
import datetime
from previewaquisator import PreviewAquisator
class TimeSequenceAquisitor(PreviewAquisator):
    # 'Constants'
        
    def __init__(self,chans, cam, shutters,_log={}):
        PreviewAquisator.__init__(self, chans, cam,shutters, None)
        #self.piezos = _piezos
        self.log = _log
        
        
        self.SeqLength = 100  
        self.TimeStep = 0.1            
    def getNextDsSlice(self):
        return self.ds.nextZ()
    def SetSeqLength(self,iLength):
        self.SeqLength = iLength
    def GetSeqLength(self):
        return self.SeqLength
    
    def SetTimeStep(self,iLength):
        self.TimeStep = iLength
    def GetTimeStep(self):
        return self.TimeStep
    
    def start(self):
        PreviewAquisator.start(self,1000*self.TimeStep)
           
    def Verify(self):
        return (True,) 
    def doStartLog(self):
        if not 'GENERAL' in self.log.keys():
            self.log['GENERAL'] = {}
        if not 'PIEZOS' in self.log.keys():
            self.log['PIEZOS'] = {}
        if not 'CAMERA' in self.log.keys():
            self.log['CAMERA'] = {}
            
        #self.log['PIEZOS']['Piezo'] = self.piezos[self.GetScanChannel()][2]
        #self.log['PIEZOS']['Stepsize'] = self.GetStepSize()
        #self.log['PIEZOS']['StartPos'] = self.GetStartPos()
        #for pz in self.piezos:
        #    self.log['PIEZOS']['%s_Pos' % pz[2]] = pz[0].GetPos(pz[1])
            
        #self.log['STEPPER']['XPos'] = m_pDoc->Step.GetNullX(),
        #m_pDoc->Step.GetPosX(), m_pDoc->Step.GetNullY(), m_pDoc->Step.GetPosY(),
        #m_pDoc->Step.GetNullZ(), m_pDoc->Step.GetPosZ());
        self.cam.GetStatus()
        self.log['CAMERA']['Binning'] = self.cam.GetHorizBin()
        #m_pDoc->LogData.SetIntegTime(m_pDoc->Camera.GetIntegTime());
        self.log['GENERAL']['Width'] = self.cam.GetPicWidth()
        self.log['GENERAL']['Height'] = self.cam.GetPicHeight()
        self.log['CAMERA']['ROIPosX'] = self.cam.GetROIX1()
        self.log['CAMERA']['ROIPosY'] = self.cam.GetROIY1()
        self.log['CAMERA']['ROIWidth'] = self.cam.GetROIX2() - self.cam.GetROIX1()
        self.log['CAMERA']['ROIHeight'] = self.cam.GetROIY2() - self.cam.GetROIY1()
        self.log['CAMERA']['StartCCDTemp'] = self.cam.GetCCDTemp()
        self.log['CAMERA']['StartElectrTemp'] = self.cam.GetElectrTemp()
        
        self.log['GENERAL']['NumChannels'] = self.ds.getNumChannels()
        self.log['GENERAL']['NumHWChans'] = self.numHWChans
        
        for ind in range(self.numHWChans):
            self.log['SHUTTER_%d' % ind] = {}
            self.log['SHUTTER_%d' % ind]['Name'] = self.chans.names[ind]
            self.log['SHUTTER_%d' % ind]['IntegrationTime'] = self.chans.itimes[ind]
            self.log['SHUTTER_%d' % ind]['Mask'] = self.hwChans[ind]
            
            s = ''
            bef = 0
            if (self.cols[ind] & self.BW):
                s = s + 'BW'
                bef = 1
            if (self.cols[ind] & self.RED):
                if bef:
                    s = s + ' '
                s = s + 'R'
                bef = 1
            if (self.cols[ind] & self.GREEN1):
                if bef:
                    s = s + ' '
                s = s + 'G1'
                bef = 1
            if (self.cols[ind] & self.GREEN2):
                if bef:
                    s = s + ' '
                s = s + 'G2'
                bef = 1
            if (self.cols[ind] & self.BLUE):
                if bef:
                    s = s + ' '
                s = s + 'B'
                #bef = 1
            self.log['SHUTTER_%d' % ind]['Colours'] = s
        
        dt = datetime.datetime.now()
        
        self.log['GENERAL']['Date'] = '%d/%d/%d' % (dt.day, dt.month, dt.year)
        self.log['GENERAL']['StartTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)
        #m_pDoc->LogData.SaveSeqROIMode(m_pDoc->Camera.GetROIMode());
        #pass
    def doStopLog(self):
        self.log['GENERAL']['Depth'] = self.ds.getDepth()
        #self.log['PIEZOS']['EndPos'] = self.GetEndPos()
        self.cam.GetStatus()
        self.log['CAMERA']['EndCCDTemp'] = self.cam.GetCCDTemp()
        self.log['CAMERA']['EndElectrTemp'] = self.cam.GetElectrTemp()
        
        dt = datetime.datetime.now()
        self.log['GENERAL']['EndTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)
  
   
