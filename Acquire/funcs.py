#!/usr/bin/python

##################
# funcs.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python
#import sys
#sys.path.append(".")

import wx
#import PYME.cSMI as example
import previewaquisator
import simplesequenceaquisator
import prevviewer
import PYME.DSView.dsviewer_npy as dsviewer
#import PYME.DSView.myviewpanel as viewpanel
import PYME.DSView.viewpanellite as viewpanel
import PYME.DSView.displaySettingsPanel as disppanel

import PYME.Acquire.protocol as protocol
from PYME.Acquire import MetaDataHandler
from PYME.Acquire.Hardware import ccdCalibrator

from PYME.cSMI import CDataStack_AsArray
from math import exp
import sqlite3
import cPickle as pickle
import os
from numpy import ndarray
import datetime
#import piezo_e662
#import piezo_e816

#teach sqlite about numpy arrays
def adapt_numarray(array):
    return sqlite3.Binary(array.dumps())

def convert_numarray(s):
    return pickle.loads(s)

sqlite3.register_adapter(ndarray, adapt_numarray)
sqlite3.register_converter("ndarray", convert_numarray)



class microscope:
    def __init__(self):
        #example.CShutterControl.init()

        #self.cam = example.CCamera()
        #self.cam.Init()

        #list of tuples  of form (class, chan, name) describing the instaled piezo channels
        self.piezos = []
        self.EnableJoystick = None

        #self.WantEventNotification = []
 
        #self.windows = []
        self.StatusCallbacks = [] #list of functions which provide status information
        self.CleanupFunctions = [] #list of functions to be called at exit
        #preview
        self.saturationThreshold = 16383 #14 bit
        self.lastFrameSaturated = False
        #self.cam.saturationIntervened = False
        self.saturatedMessage = ''

        protocol.scope = self
        ccdCalibrator.setScope(self)
        self.initDone = False

        self._OpenSettingsDB()

        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)

    def _OpenSettingsDB(self):
        #create =  not os.path.exists('PYMESettings.db')

        self.settingsDB = sqlite3.connect('PYMESettings.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        self.settingsDB.isolation_level = None

        tableNames = [a[0] for a in self.settingsDB.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]

        if not 'CCDCalibration' in tableNames:
            self.settingsDB.execute("CREATE TABLE CCDCalibration (time timestamp, temperature integer, nominalGains ndarray, trueGains ndarray)")
        if not 'VoxelSizes' in tableNames:
            self.settingsDB.execute("CREATE TABLE VoxelSizes (ID INTEGER PRIMARY KEY, x REAL, y REAL, name TEXT)")
        if not 'VoxelSizeHistory' in tableNames:
            self.settingsDB.execute("CREATE TABLE VoxelSizeHistory (time timestamp, sizeID INTEGER)")
        if not 'StartupTimes' in tableNames:
            self.settingsDB.execute("CREATE TABLE StartupTimes (component TEXT, time REAL)")
            self.settingsDB.execute("INSERT INTO StartupTimes VALUES ('total', 5)")
            
        self.settingsDB.commit()

    def GenStartMetadata(self, mdh):
        currVoxelSizeID = self.settingsDB.execute("SELECT sizeID FROM VoxelSizeHistory ORDER BY time DESC").fetchone()
        if not currVoxelSizeID == None:
            voxx, voxy = self.settingsDB.execute("SELECT x,y FROM VoxelSizes WHERE ID=?", currVoxelSizeID).fetchone()
            mdh.setEntry('voxelsize.x', voxx)
            mdh.setEntry('voxelsize.y', voxy)
            mdh.setEntry('voxelsize.units', 'um')

        for p in self.piezos:
            mdh.setEntry('Positioning.%s' % p[2].replace(' ', '_').replace('-', '_'), p[0].GetPos(p[1]))

    def AddVoxelSizeSetting(self, name, x, y):
        self.settingsDB.execute("INSERT INTO VoxelSizes (name, x, y) VALUES (?, ?, ?)", (name, x, y))
        self.settingsDB.commit()
        

    def SetVoxelSize(self, voxelsizename):
        voxelSizeID = self.settingsDB.execute("SELECT ID FROM VoxelSizes WHERE name=?", (voxelsizename,)).fetchone()[0]
        self.settingsDB.execute("INSERT INTO VoxelSizeHistory VALUES (?, ?)", (datetime.datetime.now(), voxelSizeID))
        self.settingsDB.commit()

    def pr_refr(self, source):
        self.prev_fr.update()
        
    def pr_refr2(self, source):
        self.vp.imagepanel.Refresh()

    def satCheck(self, source): # check for saturation
        im = CDataStack_AsArray(source.ds, 0)
        IMax = im.max()

        if not self.cam.shutterOpen:
            self.cam.ADOffset = im.mean()
        elif (IMax >= self.cam.SaturationThreshold): #is saturated

            source.cam.StopAq()


            if self.lastFrameSaturated: #last frame was also saturated - our intervention obviously didn't work - close the shutter
                if 'SetShutter' in dir(source.cam):
                    source.cam.SetShutter(False)
                source.cam.StartExposure()
                self.saturatedMessage = 'Sat - Camera shutter was closed'
                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                return

            fracPixelsSat = (im > self.saturationThreshold).sum().astype('f')/im.size

            #try turning the e.m. gain off
            if 'SetEMGain' in dir(source.cam) and not source.cam.GetEMGain() == 0:
                self.oldEMGain = source.cam.GetEMGain()
                source.cam.SetEMGain(0)
                if self.oldEMGain  < 50: #poor chance of resolving by turning EMGain down alone
                    if 'SetShutter' in dir(source.cam):
                        source.cam.SetShutter(False)
                        self.saturatedMessage = 'Sat - Camera shutter was closed'
                else:
                    self.saturatedMessage = 'Sat - EM Gain turned down'
                    
                source.cam.StartExposure()

                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                return
            else:
                if 'SetShutter' in dir(source.cam):
                    source.cam.SetShutter(False)
                source.cam.StartExposure()
                self.saturatedMessage = 'Sat - Camera shutter was closed'
                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                return

            self.lastFrameSaturated = True

        else:
            self.lastFrameSaturated = False






    def genStatus(self):
        stext = ''
        if self.cam.CamReady():
            self.cam.GetStatus()
            stext = 'CCD Temp: %d' % self.cam.GetCCDTemp()
        else:
            stext = '<Camera ERROR>'
        if 'saturationIntervened' in dir(self.cam):
            if self.lastFrameSaturated:
                stext = stext + '    Camera Saturated!!'
            if self.cam.saturationIntervened:
                stext = stext + '    ' + self.saturatedMessage
        if 'step' in dir(self):
            stext = stext + '   Stepper: (XPos: %1.2f  YPos: %1.2f  ZPos: %1.2f)' % (self.step.GetPosX(), self.step.GetPosY(), self.step.GetPosZ())
        if self.pa.isRunning():
            if 'GetFPS' in dir(self.cam):
                stext = stext + '    FPS = (%2.2f/%2.2f)' % (self.cam.GetFPS(),self.pa.getFPS())
            else:
                stext = stext + '    FPS = %2.2f' % self.pa.getFPS()

            if 'GetNumImsBuffered' in dir(self.cam):
				stext = stext + '    Buffer Level: %d of %d' % (self.cam.GetNumImsBuffered(), self.cam.GetBufferSize())
        
        for sic in self.StatusCallbacks:
                stext = stext + '    ' + sic()       
        return stext

    def livepreview(self, Parent=None, Notebook = None):
        self.pa = previewaquisator.PreviewAquisator(self.chaninfo,self.cam, self.shutters)
        self.pa.Prepare()
        
        if (Notebook == None):
            self.prev_fr = prevviewer.PrevViewFrame(Parent, "Live Preview", self.pa.ds)
            self.pa.WantFrameGroupNotification.append(self.pr_refr)
            self.prev_fr.genStatusText = self.genStatus
            self.prev_fr.Show()
        else:
             self.vp = viewpanel.MyViewPanel(Notebook, self.pa.ds)
             self.vp.crosshairs = False

             self.vsp = disppanel.dispSettingsPanel(Notebook, self.vp)

             self.pa.WantFrameGroupNotification.append(self.pr_refr2)
             if 'shutterOpen' in dir(self.cam):
                self.pa.WantFrameGroupNotification.append(self.satCheck)
             Parent.time1.WantNotification.append(self.vsp.RefrData)
             #Notebook.AddPage(imageId=-1, page=self.vp, select=True,text='Preview')
             Notebook.AddPage(page=self.vp, select=True,caption='Preview')
             #Notebook._mgr.AddPane

             Parent.AddCamTool(self.vsp, 'Display')
#             Notebook.AddPage(page=self.vsp, select=False,caption='Display')
#             Notebook.Split(3, wx.RIGHT)
#             Notebook.SetSelection(2)
#             Notebook.SetSelection(3)
             
        self.pa.start()

    #aquisition

    def aq_refr(self,source):
        if not self.pb.Update(self.sa.ds.getZPos(), 'Slice %d of %d' % (self.sa.ds.getZPos(), self.sa.ds.getDepth())):
            self.sa.stop()
        #self.dfr.update()

    def aq_end(self,source):
        #self.dfr.update()
        self.pb.Update(self.sa.ds.getDepth())
        
        if 'step' in self.__dict__:
            self.sa.log['STEPPER'] = {}
            self.sa.log['STEPPER']['XPos'] = self.step.GetPosX()
            self.sa.log['STEPPER']['YPos'] = self.step.GetPosY()
            self.sa.log['STEPPER']['ZPos'] = self.step.GetPosZ()
            
        if 'scopedetails' in self.__dict__:
            self.sa.log['MICROSCOPE'] = self.scopedetails
        
        dialog = wx.MessageDialog(None, "Aquisition Finished", "pySMI", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()
        
        self.pa.Prepare(True)
        self.pa.start()

        self.dfr = dsviewer.DSViewFrame(None, "New Aquisition", CDataStack_AsArray(self.sa.ds, 0), self.sa.log, mdh=self.sa.mdh)
        self.dfr.Show()
        self.sa.ds = None

    def aqt_refr(self,source):
        if not self.pb.Update(self.ta.ds.getZPos(), 'Slice %d of %d' % (self.ta.ds.getZPos(), self.ta.ds.getDepth())):
            self.ta.stop()
        #self.dfr.update()

    def aqt_end(self,source):
        #self.dfr.update()
        self.pb.Update(self.ta.ds.getDepth())
        
        if 'step' in self.__dict__:
            self.sa.log['STEPPER'] = {}
            self.sa.log['STEPPER']['XPos'] = self.step.GetPosX()
            self.sa.log['STEPPER']['YPos'] = self.step.GetPosY()
            self.sa.log['STEPPER']['ZPos'] = self.step.GetPosZ()
            
        if 'scopedetails' in self.__dict__:
            self.ta.log['MICROSCOPE'] = self.scopedetails
        
        dialog = wx.MessageDialog(None, "Aquisition Finished", "pySMI", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()
        
        self.pa.Prepare(True)
        self.pa.start()

        self.dfr = dsviewer.DSViewFrame(None, "New Aquisition", CDataSatck_AsArray(self.ta.ds, 0), self.ta.log)
        self.dfr.Show()
        self.ta.ds = None

    def aquireStack(self,piezo, startpos, endpos, stepsize, channel = 1):
        self.pa.stop()

        self.sa = simplesequenceaquisator.SimpleSequenceAquisitor(self.chaninfo, self.cam, self.shutters, piezo)
        self.sa.SetStartMode(self.sa.START_AND_END)

        self.sa.SetStepSize(stepsize)
        self.sa.SetStartPos(endpos)
        self.sa.SetEndPos(startpos)

        self.sa.Prepare()

        
        self.sa.WantFrameNotification.append(self.aq_refr)

        self.sa.WantStopNotification.append(self.aq_end)

        self.sa.start()

        self.pb = wx.ProgressDialog('Aquisition in progress ...', 'Slice 1 of %d' % self.sa.ds.getDepth(), self.sa.ds.getDepth(), style = wx.PD_APP_MODAL|wx.PD_AUTO_HIDE|wx.PD_REMAINING_TIME|wx.PD_CAN_ABORT)

    def turnAllLasersOff(self):
        for l in self.lasers:
            l.TurnOff()

    def __del__(self):
        self.settingsDB.close()
