import os
#import logparser
import datetime
import tables
from PYME import cSMI

import time

import eventLog

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

class EventLogger:
   def __init__(self, scope, hdf5File):
      self.scope = scope
      self.hdf5File = hdf5File

      self.evts = self.hdf5File.createTable(hdf5File.root, 'Events', SpoolEvent)

   def logEvent(self, eventName, eventDescr = ''):
      ev = self.evts.row

      ev['EventName'] = eventName
      ev['EventDescr'] = eventDescr
      ev['Time'] = time.time()

      ev.append()
      self.evts.flush()

class Spooler:
   def __init__(self, scope, filename, acquisator, parent=None, complevel=6, complib='zlib'):
       self.scope = scope
       self.filename=filename
       self.acq = acquisator
       self.parent = parent 
       
       #self.dirname =filename[:-4]
       #os.mkdir(self.dirname)
       
       #self.filestub = self.dirname.split(os.sep)[-1]
       
       self.h5File = tables.openFile(filename, 'w')
       
       filt = tables.Filters(complevel, complib, shuffle=True)

       self.imageData = self.h5File.createEArray(self.h5File.root, 'ImageData', tables.UInt16Atom(), (0,scope.cam.GetPicWidth(),scope.cam.GetPicHeight()), filters=filt)

       self.imNum=0
       self.log = {}
       self.doStartLog()

       self.evtLogger = EventLogger(scope, self.h5File)
       eventLog.WantEventNotification.append(self.evtLogger)
       
       self.acq.WantFrameNotification.append(self.Tick)
       
       self.spoolOn = True
       
   def StopSpool(self):
       self.acq.WantFrameNotification.remove(self.Tick)
       eventLog.WantEventNotification.remove(self.evtLogger)
       self.doStopLog()
       #self.writeLog()
       self.h5File.flush()
       self.h5File.close()
       self.spoolOn = False
   
   def Tick(self, caller):
      #fn = self.dirname + os.sep + self.filestub +'%05d.kdf' % self.imNum
      #caller.ds.SaveToFile(fn.encode())
      
      self.imageData.append(cSMI.CDataStack_AsArray(caller.ds, 0).reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()))
      self.h5File.flush()

      self.imNum += 1
      if not self.parent == None:
         self.parent.Tick()

   def doStartLog(self):
      md = self.h5File.createGroup(self.h5File.root, 'MetaData')

      dt = datetime.datetime.now()
        
      self.dtStart = dt

      #self.log['GENERAL']['Date'] = '%d/%d/%d' % (dt.day, dt.month, dt.year)
      #self.log['GENERAL']['StartTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)
      md._v_attrs.StartTime = time.time()
      
      self.h5File.createGroup(self.h5File.root.MetaData, 'Camera')
      self.scope.cam.GetStatus()

      if 'tKin' in dir(self.scope.cam): #check for Andor cam
         md.Camera._v_attrs.IntegrationTime = self.scope.cam.tExp
         md.Camera._v_attrs.CycleTime = self.scope.cam.tKin
         md.Camera._v_attrs.EMGain = self.scope.cam.GetEMGain()

      md.Camera._v_attrs.ROIPosX = self.scope.cam.GetROIX1()
      md.Camera._v_attrs.ROIPosY = self.scope.cam.GetROIY1()
      md.Camera._v_attrs.StartCCDTemp = self.scope.cam.GetCCDTemp()
      
   
   def doStartLoog(self):
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
        #self.log['CAMERA']['EndCCDTemp'] = self.scope.cam.GetCCDTemp()
        #self.log['CAMERA']['EndElectrTemp'] = self.scope.cam.GetElectrTemp()
        
        dt = datetime.datetime.now()
        #self.log['GENERAL']['EndTime'] = '%d:%d:%d' % (dt.hour, dt.minute, dt.second)
        self.h5File.root.MetaData._v_attrs.EndTime = time.time()
        #self.log['GENERAL']['NumImages'] = '%d' % self.imNum   
        
   def writeLog_(self):
        lw = logparser.logwriter()
        s = lw.write(self.log)
        log_f = file(self.filename, 'w')
        log_f.write(s)
        log_f.close()
        
   def __del__(self):
        if self.spoolOn:
            self.StopSpool()
