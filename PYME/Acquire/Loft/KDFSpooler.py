#!/usr/bin/python

##################
# KDFSpooler.py
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

import os
import logparser
import datetime

class Spooler:
   def __init__(self, scope, filename, acquisator, parent=None):
       self.scope = scope
       self.filename=filename
       self.acq = acquisator
       self.parent = parent 
       
       self.dirname =filename[:-4]
       os.mkdir(self.dirname)
       
       self.filestub = self.dirname.split(os.sep)[-1]
       
       self.imNum=0
       self.log = {}
       self.doStartLog()
       
       self.acq.WantFrameNotification.append(self.Tick)
       
       self.spoolOn = True
       
   def StopSpool(self):
       self.acq.WantFrameNotification.remove(self.Tick)
       self.doStopLog()
       self.writeLog()
       self.spoolOn = False
   
   def Tick(self, caller):
       fn = self.dirname + os.sep + self.filestub +'%05d.kdf' % self.imNum
       caller.ds.SaveToFile(fn.encode())
       self.imNum += 1
       if not self.parent is None:
           self.parent.Tick()
   
   def doStartLog(self):
        if not 'GENERAL' in self.log.keys():
            self.log['GENERAL'] = {}
        if not 'PIEZOS' in self.log.keys():
            self.log['PIEZOS'] = {}
        if not 'CAMERA' in self.log.keys():
            self.log['CAMERA'] = {}
            
        for pz in self.scope.piezos:
            self.log['PIEZOS']['%s_Pos' % pz[2]] = pz[0].GetPos(pz[1])
            
        #self.log['STEPPER']['XPos'] = m_pDoc->Step.GetNullX(),
        #m_pDoc->Step.GetPosX(), m_pDoc->Step.GetNullY(), m_pDoc->Step.GetPosY(),
        #m_pDoc->Step.GetNullZ(), m_pDoc->Step.GetPosZ());
        self.scope.cam.GetStatus()
        self.log['CAMERA']['Binning'] = self.scope.cam.GetHorizBin()
        #m_pDoc->LogData.SetIntegTime(m_pDoc->Camera.GetIntegTime());
        if 'tKin' in dir(self.scope.cam): #check for Andor cam
           self.log['CAMERA']['IntegrationTime'] = self.scope.cam.tExp
           self.log['CAMERA']['CycleTime'] = self.scope.cam.tKin
           self.log['CAMERA']['EMGain'] = self.scope.cam.GetEMGain()
        self.log['GENERAL']['Width'] = self.scope.cam.GetPicWidth()
        self.log['GENERAL']['Height'] = self.scope.cam.GetPicHeight()
        self.log['CAMERA']['ROIPosX'] = self.scope.cam.GetROIX1()
        self.log['CAMERA']['ROIPosY'] = self.scope.cam.GetROIY1()
        self.log['CAMERA']['ROIWidth'] = self.scope.cam.GetROIX2() - self.scope.cam.GetROIX1()
        self.log['CAMERA']['ROIHeight'] = self.scope.cam.GetROIY2() - self.scope.cam.GetROIY1()
        self.log['CAMERA']['StartCCDTemp'] = self.scope.cam.GetCCDTemp()
        self.log['CAMERA']['StartElectrTemp'] = self.scope.cam.GetElectrTemp()
        
        #self.log['GENERAL']['NumChannels'] = self.ds.getNumChannels()
        #self.log['GENERAL']['NumHWChans'] = self.numHWChans
        
        #for ind in range(self.numHWChans):
        #    self.log['SHUTTER_%d' % ind] = {}
        #    self.log['SHUTTER_%d' % ind]['Name'] = self.chans.names[ind]
        #    self.log['SHUTTER_%d' % ind]['IntegrationTime'] = self.chans.itimes[ind]
        #    self.log['SHUTTER_%d' % ind]['Mask'] = self.hwChans[ind]
            
        #    s = ''
        ##            bef = 0
        ##            if (self.cols[ind] & self.BW):
        ##                s = s + 'BW'
        ##                bef = 1
        ##            if (self.cols[ind] & self.RED):
        ##                if bef:
        ##                    s = s + ' '
        ##                s = s + 'R'
        ##                bef = 1
        ##            if (self.cols[ind] & self.GREEN1):
        ##                if bef:
        ##                    s = s + ' '
        ##                s = s + 'G1'
        ##                bef = 1
        ##            if (self.cols[ind] & self.GREEN2):
        ##                if bef:
        ##                    s = s + ' '
        ##                s = s + 'G2'
        ##                bef = 1
        ##            if (self.cols[ind] & self.BLUE):
        ##                if bef:
        ##                    s = s + ' '
        ##                s = s + 'B'
        ##                #bef = 1
        ##            self.log['SHUTTER_%d' % ind]['Colours'] = s
        
        dt = datetime.datetime.now()
        
        self.dtStart = dt
        
        self.log['GENERAL']['Date'] = '%d/%d/%d' % (dt.day, dt.month, dt.year)
        self.log['GENERAL']['StartTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)

   def doStopLog(self):
        #self.log['GENERAL']['Depth'] = self.ds.getDepth()
        #self.log['PIEZOS']['EndPos'] = self.GetEndPos()
        self.scope.cam.GetStatus()
        self.log['CAMERA']['EndCCDTemp'] = self.scope.cam.GetCCDTemp()
        self.log['CAMERA']['EndElectrTemp'] = self.scope.cam.GetElectrTemp()
        
        dt = datetime.datetime.now()
        self.log['GENERAL']['EndTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second) 
        self.log['GENERAL']['NumImages'] = '%d' % self.imNum   
        
   def writeLog(self):
        lw = logparser.logwriter()
        s = lw.write(self.log)
        log_f = open(self.filename, 'w')
        log_f.write(s)
        log_f.close()
        
   def __del__(self):
        if self.spoolOn:
            self.StopSpool()
