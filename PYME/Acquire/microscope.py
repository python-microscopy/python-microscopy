#!/usr/bin/python

##################
# funcs.py
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

#!/usr/bin/python
#import sys
#sys.path.append(".")

import wx
#import PYME.cSMI as example
from PYME.Acquire import previewaquisator as previewaquisator
from PYME.Acquire import simplesequenceaquisator
from PYME.Acquire import prevviewer
import PYME.DSView.dsviewer_npy_nb as dsviewer
#import PYME.DSView.myviewpanel as viewpanel
import PYME.DSView.viewpanellite as viewpanel
import PYME.DSView.displaySettingsPanel as disppanel
from PYME.DSView import arrayViewPanel

import PYME.Acquire.protocol as protocol
from PYME.Acquire import MetaDataHandler
from PYME.Acquire.Hardware import ccdCalibrator

from PYME.cSMI import CDataStack_AsArray
from math import exp
import sqlite3
#import cPickle as pickle
import os
#from numpy import ndarray
import datetime
#import piezo_e662
#import piezo_e816

#register handlers for ndarrays
from PYME.misc import sqlitendarray
#from PYME.Acquire import MetaDataHandler


class microscope(object):
    def __init__(self):
        #list of tuples  of form (class, chan, name) describing the instaled piezo channels
        self.piezos = []
        self.hardwareChecks = []
        
        #entries should be of the form: "x" : (piezo, channel, multiplier)
        # where multiplyier is what to multiply by to get the usints to um
        self.positioning = {}
        self.joystick = None

        self.cameras = {}
        self.camControls = {}

        self.stackNum = 0

        #self.WantEventNotification = []
 
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
        
        #provision to set global metadata values in startup script
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        
    def GetPos(self):
        res = {}
        for k in self.positioning.keys():
            p, c, m = self.positioning[k]
            res[k] = p.GetPos(c)*m
            
        return res
        
    def SetPos(self, **kwargs):
        for k, v in kwargs.items():
            p, c, m = self.positioning[k]
            p.MoveTo(c, v/m)
        

    def _OpenSettingsDB(self):
        #create =  not os.path.exists('PYMESettings.db')
        fstub = os.path.split(__file__)[0]
        dbfname = os.path.join(fstub, 'PYMESettings.db')

        self.settingsDB = sqlite3.connect(dbfname, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        self.settingsDB.isolation_level = None

        tableNames = [a[0] for a in self.settingsDB.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]

        if not 'CCDCalibration2' in tableNames:
            self.settingsDB.execute("CREATE TABLE CCDCalibration2 (time timestamp, temperature integer, serial integer, nominalGains ndarray, trueGains ndarray)")
        if not 'VoxelSizes' in tableNames:
            self.settingsDB.execute("CREATE TABLE VoxelSizes (ID INTEGER PRIMARY KEY, x REAL, y REAL, name TEXT)")
        if not 'VoxelSizeHistory2' in tableNames:
            self.settingsDB.execute("CREATE TABLE VoxelSizeHistory2 (time timestamp, sizeID INTEGER, camSerial INTEGER)")
        if not 'StartupTimes' in tableNames:
            self.settingsDB.execute("CREATE TABLE StartupTimes (component TEXT, time REAL)")
            self.settingsDB.execute("INSERT INTO StartupTimes VALUES ('total', 5)")
            
        self.settingsDB.commit()

    def GetPixelSize(self):
        currVoxelSizeID = self.settingsDB.execute("SELECT sizeID FROM VoxelSizeHistory2 WHERE camSerial=? ORDER BY time DESC", (self.cam.GetSerialNumber(),)).fetchone()
        if not currVoxelSizeID == None:
            return self.settingsDB.execute("SELECT x,y FROM VoxelSizes WHERE ID=?", currVoxelSizeID).fetchone()

    def GenStartMetadata(self, mdh):
        currVoxelSizeID = self.settingsDB.execute("SELECT sizeID FROM VoxelSizeHistory2 WHERE camSerial=? ORDER BY time DESC", (self.cam.GetSerialNumber(),)).fetchone()
        if not currVoxelSizeID == None:
            voxx, voxy = self.settingsDB.execute("SELECT x,y FROM VoxelSizes WHERE ID=?", currVoxelSizeID).fetchone()
            mdh.setEntry('voxelsize.x', voxx)
            mdh.setEntry('voxelsize.y', voxy)
            mdh.setEntry('voxelsize.units', 'um')

        for p in self.piezos:
            mdh.setEntry('Positioning.%s' % p[2].replace(' ', '_').replace('-', '_'), p[0].GetPos(p[1]))
            
        mdh.copyEntriesFrom(self.mdh)

    def AddVoxelSizeSetting(self, name, x, y):
        self.settingsDB.execute("INSERT INTO VoxelSizes (name, x, y) VALUES (?, ?, ?)", (name, x, y))
        self.settingsDB.commit()
        

    def SetVoxelSize(self, voxelsizename, camName=None):
        if camName == None:
            cam = self.cam
        else:
            cam = self.cameras[camName]
            
        voxelSizeID = self.settingsDB.execute("SELECT ID FROM VoxelSizes WHERE name=?", (voxelsizename,)).fetchone()[0]
        self.settingsDB.execute("INSERT INTO VoxelSizeHistory2 VALUES (?, ?, ?)", (datetime.datetime.now(), voxelSizeID, cam.GetSerialNumber()))
        self.settingsDB.commit()

    def pr_refr(self, source):
        self.prev_fr.update()
        
#    def pr_refr2(self, source):
#        #self.vp.imagepanel.Refresh()
#        self.vp.Redraw()

    def pr_refr3(self, souce):
        self.sp.refr()

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
                self.saturatedMessage = 'Camera shutter has been closed'
                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                wx.MessageBox(self.saturatedMessage, "Saturation detected", wx.OK|wx.ICON_HAND)
                return

            fracPixelsSat = (im > self.saturationThreshold).sum().astype('f')/im.size

            #try turning the e.m. gain off
            if 'SetEMGain' in dir(source.cam) and not source.cam.GetEMGain() == 0:
                self.oldEMGain = source.cam.GetEMGain()
                source.cam.SetEMGain(0)
                if self.oldEMGain  < 50: #poor chance of resolving by turning EMGain down alone
                    if 'SetShutter' in dir(source.cam):
                        source.cam.SetShutter(False)
                        self.saturatedMessage = 'Camera shutter closed'
                else:
                    self.saturatedMessage = 'EM Gain turned down'
                    
                source.cam.StartExposure()

                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                wx.MessageBox(self.saturatedMessage, "Saturation detected", wx.OK|wx.ICON_HAND)
                return
            else:
                if 'SetShutter' in dir(source.cam):
                    source.cam.SetShutter(False)
                source.cam.StartExposure()
                self.saturatedMessage = 'Camera shutter closed'
                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                wx.MessageBox(self.saturatedMessage, "Saturation detected", wx.OK|wx.ICON_HAND)
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
#        if 'pa' in dir(self):
#            self.pa.stop() #stop old acquisition

        self.pa = previewaquisator.PreviewAquisator(self.chaninfo,self.cam, self.shutters)
        self.pa.HardwareChecks.extend(self.hardwareChecks)
        self.pa.Prepare()

        if self.cam.GetPicHeight() > 1:
            if 'vp' in dir(self):
                    self.vp.SetDataStack(self.pa.dsa)
            elif (Notebook == None):
                self.prev_fr = prevviewer.PrevViewFrame(Parent, "Live Preview", self.pa.dsa)
                self.pa.WantFrameGroupNotification.append(self.pr_refr)
                self.prev_fr.genStatusText = self.genStatus
                self.prev_fr.Show()
            else:
                self.vp = arrayViewPanel.ArrayViewPanel(Notebook, self.pa.dsa)
                self.vp.crosshairs = False
                self.vp.DrawScaleBar = False
                self.vp.do.leftButtonAction = self.vp.do.ACTION_SELECTION
                self.vp.do.showSelection = True
                self.vp.CenteringHandlers.append(self.centreView)

                self.vsp = disppanel.dispSettingsPanel2(Notebook, self.vp)


                Parent.time1.WantNotification.append(self.vsp.RefrData)

                Notebook.AddPage(page=self.vp, select=True,caption='Preview')

                Parent.AddCamTool(self.vsp, 'Display')

            #self.pa.WantFrameGroupNotification.append(self.pr_refr2)
            self.pa.WantFrameGroupNotification.append(self.vp.Redraw)

        else:
            #1d data - use graph instead
            from PYME.Analysis.LMVis import fastGraph
            if 'sp' in dir(self):
                    pass
            elif (Notebook == None):
                self.prev_fr = prevviewer.PrevViewFrame(Parent, "Preview", self.pa.ds)
                self.pa.WantFrameGroupNotification.append(self.pr_refr)
                self.prev_fr.genStatusText = self.genStatus
                self.prev_fr.Show()
            else:
                self.sp = fastGraph.SpecGraphPanel(Notebook, self)

                Notebook.AddPage(page=self.sp, select=True,caption='Preview')

            self.pa.WantFrameGroupNotification.append(self.pr_refr3)

        if 'shutterOpen' in dir(self.cam):
            self.pa.WantFrameGroupNotification.append(self.satCheck)
            
        self.pa.start()

    def SetCamera(self, camName):
        if 'pa' in dir(self):
            self.pa.stop()

        #deactivate cameras
        for c in self.cameras.values():
            c.SetActive(False)
            c.SetShutter(False)
            
        for k in self.cameras.keys():
            self.camControls[k].GetParent().Hide()#GetParent().UnPin()
        
        self.cam = self.cameras[camName]
        if 'lightpath' in dir(self):
            self.lightpath.SetPort(self.cam.port)
        
        self.cam.SetActive(True)
        self.cam.SetShutter(self.camControls[camName].cbShutter.GetValue())
        self.camControls[camName].GetParent().Show()#GetParent().PinOpen()
        self.camControls[camName].GetParent().GetParent().Layout()

        if 'sa' in dir(self):
            self.sa.cam = self.cam

        if 'pa' in dir(self):
            self.livepreview()
            
        
            
    def centreView(self, dx, dy):
        vx, vy = self.GetPixelSize()
        
        p = self.GetPos()
        
        ox = p['x']
        oy = p['y']
        
        #print dx, dy, vx, vy, dx*vx
        
        self.SetPos(x=(ox + dx*vx), y=(oy + dy*vy))


    #aquisition

#    def aq_refr(self,source):
#        if not self.pb.Update(self.sa.ds.getZPos(), 'Slice %d of %d' % (self.sa.ds.getZPos(), self.sa.ds.getDepth())):
#            self.sa.stop()
#        #self.dfr.update()

#    def aq_end(self,source):
#        #self.dfr.update()
#        self.pb.Update(self.sa.ds.getDepth())
#        
#        if 'step' in self.__dict__:
#            self.sa.log['STEPPER'] = {}
#            self.sa.log['STEPPER']['XPos'] = self.step.GetPosX()
#            self.sa.log['STEPPER']['YPos'] = self.step.GetPosY()
#            self.sa.log['STEPPER']['ZPos'] = self.step.GetPosZ()
#            
#        if 'scopedetails' in self.__dict__:
#            self.sa.log['MICROSCOPE'] = self.scopedetails
#        
#        dialog = wx.MessageDialog(None, "Aquisition Finished", "pySMI", wx.OK)
#        dialog.ShowModal()
#        dialog.Destroy()
#        
#        self.pa.Prepare(True)
#        self.pa.start()
#
#        #self.dfr = dsviewer.DSViewFrame(None, "New Aquisition", CDataStack_AsArray(self.sa.ds, 0).squeeze(), self.sa.log, mdh=self.sa.mdh)
#        #self.dfr.Show()
#
#        im = dsviewer.ImageStack(data = CDataStack_AsArray(self.sa.ds, 0).squeeze(), mdh = self.sa.mdh)
#        dvf = dsviewer.DSViewFrame(im, title=('<Unsaved Stack %d>' % self.stackNum), mode='lite', size=(500, 500))
#        dvf.SetSize((500,500))
#        dvf.Show()
#
#        self.stackNum +=1
#
#        self.sa.ds = None

#    def aqt_refr(self,source):
#        if not self.pb.Update(self.ta.ds.getZPos(), 'Slice %d of %d' % (self.ta.ds.getZPos(), self.ta.ds.getDepth())):
#            self.ta.stop()
#        #self.dfr.update()

#    def aqt_end(self,source):
#        #self.dfr.update()
#        self.pb.Update(self.ta.ds.getDepth())
#        
#        if 'step' in self.__dict__:
#            self.sa.log['STEPPER'] = {}
#            self.sa.log['STEPPER']['XPos'] = self.step.GetPosX()
#            self.sa.log['STEPPER']['YPos'] = self.step.GetPosY()
#            self.sa.log['STEPPER']['ZPos'] = self.step.GetPosZ()
#            
#        if 'scopedetails' in self.__dict__:
#            self.ta.log['MICROSCOPE'] = self.scopedetails
#        
#        dialog = wx.MessageDialog(None, "Aquisition Finished", "pySMI", wx.OK)
#        dialog.ShowModal()
#        dialog.Destroy()
#        
#        self.pa.Prepare(True)
#        self.pa.start()
#
#        self.dfr = dsviewer.DSViewFrame(None, "New Aquisition", CDataSatck_AsArray(self.ta.ds, 0), self.ta.log)
#        self.dfr.Show()
#        self.ta.ds = None

#    def aquireStack(self,piezo, startpos, endpos, stepsize, channel = 1):
#        self.pa.stop()
#
#        self.sa = simplesequenceaquisator.SimpleSequenceAquisitor(self.chaninfo, self.cam, self.shutters, piezo)
#        self.sa.SetStartMode(self.sa.START_AND_END)
#
#        self.sa.SetStepSize(stepsize)
#        self.sa.SetStartPos(endpos)
#        self.sa.SetEndPos(startpos)
#
#        self.sa.Prepare()
#
#        
#        self.sa.WantFrameNotification.append(self.aq_refr)
#
#        self.sa.WantStopNotification.append(self.aq_end)
#
#        self.sa.start()
#
#        self.pb = wx.ProgressDialog('Aquisition in progress ...', 'Slice 1 of %d' % self.sa.ds.getDepth(), self.sa.ds.getDepth(), style = wx.PD_APP_MODAL|wx.PD_AUTO_HIDE|wx.PD_REMAINING_TIME|wx.PD_CAN_ABORT)

    def turnAllLasersOff(self):
        for l in self.lasers:
            l.TurnOff()

    def __del__(self):
        self.settingsDB.close()
