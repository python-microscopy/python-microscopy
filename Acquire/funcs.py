#!/usr/bin/python
#import sys
#sys.path.append(".")

import wx
#import PYME.cSMI as example
import previewaquisator
import simplesequenceaquisator
import prevviewer
import PYME.DSView.dsviewer as dsviewer
import PYME.DSView.myviewpanel as myviewpanel
#import piezo_e662
#import piezo_e816

class microscope:
    def __init__(self):
        #example.CShutterControl.init()

        #self.cam = example.CCamera()
        #self.cam.Init()

        #list of tuples  of form (class, chan, name) describing the instaled piezo channels
        self.piezos = []

        #self.WantEventNotification = []
 
        #self.windows = []

    #preview
    
    

    def pr_refr(self, source):
        self.prev_fr.update()
        
    def pr_refr2(self, source):
        self.vp.imagepanel.Refresh()

    def genStatus(self):
        stext = ''
        if self.cam.CamReady():
            self.cam.GetStatus()
            stext = 'CCD Temp: %d   Electro Temp: %d' % (self.cam.GetCCDTemp(), self.cam.GetElectrTemp())
        else:
            stext = '<Camera ERROR>'
        if 'step' in dir(self):
            stext = stext + '   Stepper: (XPos: %1.2f  YPos: %1.2f  ZPos: %1.2f)' % (self.step.GetPosX(), self.step.GetPosY(), self.step.GetPosZ())
        if self.pa.isRunning():
            if 'GetFPS' in dir(self.cam):
                stext = stext + '    FPS = (%2.2f/%2.2f)' % (self.cam.GetFPS(),self.pa.getFPS())
            else:
                stext = stext + '    FPS = %2.2f' % self.pa.getFPS()

            if 'GetBufferSize' in dir(self.cam):
				stext = stext + '    Buffer Level: %d of %d' % (self.cam.GetNumImsBuffered(), self.cam.GetBufferSize())
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
             self.vp = myviewpanel.MyViewPanel(Notebook, self.pa.ds)
             self.vp.crosshairs = False
             self.pa.WantFrameGroupNotification.append(self.pr_refr2)
             #Notebook.AddPage(imageId=-1, page=self.vp, select=True,text='Preview')
             Notebook.AddPage(page=self.vp, select=True,caption='Preview')
             
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

        self.dfr = dsviewer.DSViewFrame(None, "New Aquisition", self.sa.ds, self.sa.log) 
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

        self.dfr = dsviewer.DSViewFrame(None, "New Aquisition", self.ta.ds, self.ta.log) 
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
