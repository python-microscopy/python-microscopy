import wx
import time
import Pyro.core

class FocusCorrector(wx.Timer):
    def __init__(self, piezo, tolerance=0.2, estSlopeDyn=False, recDrift=False, axis='X', guideLaser=None):
        wx.Timer.__init__(self)
        
        self.piezo = piezo
        self.COIi = Pyro.core.getProxyForURI('PYRONAME://COIi')
        
        self.tolerance = tolerance #set a position tolerance (default 200nm)
        
        self.Tracking = False #we're not locked at the start
        self.SlopeEst = 0.5 #pretty arbitray - 200nm/pixel - will be refined as algorithm runs
        self.TargetPos = None
        
        self.EstimateSlope = estSlopeDyn

        self.recDrift = recDrift

        self.Errors = []
        self.PiezoPoss = []
        self.slopeEsts = []

        self.guideLaser = guideLaser

        if (axis == 'X'):
            self.posFcn = self.COIi.GetCOIX
        else:
            self.posFcn = self.COIi.GetCOIY
        
    def TrackOn(self, reestimateSlope=True):
        self.TargetPos = self.posFcn() #just use the vertical deflection for now
        
        if (not 'SlopeEst' in dir(self)) or reestimateSlope:
            self.DoSlopeEst()
        
        #fudge a previous correction to make our automatic slope correction work
        self.LastPos = self.TargetPos - self.SlopeEst
        self.LastStep = 1.0
        
        self.Tracking = True

        self.Errors = []
        self.PiezoPoss = []
        self.slopeEsts = []
	
        
    def TrackOff(self):
        self.Tracking = False

    def _IsGuided(self): #is the laser being used for the position sensing on? 
        if  not self.guideLaser == None:
            return self.guideLaser.IsOn()
        else:
            return True #assume that laser is on if we don't have any info one way of the other
        
    def Notify(self):
        if self.Tracking and self._IsGuided():
            currPos = self.posFcn()
            if self.EstimateSlope:
                SlopeEst_new = (self.LastPos - currPos)/self.LastStep #estimate slope
            
                self.SlopeEst = .5*self.SlopeEst + 0.5*SlopeEst_new #average slope changes
                #print self.SlopeEst
            
            posErr = currPos - self.TargetPos
            #print posErr
            if self.recDrift:
                self.Errors.append(posErr)
                self.PiezoPoss.append(self.piezo.GetPos(0))
                self.slopeEsts.append(self.SlopeEst)
            
            if abs(posErr) > abs(self.tolerance*self.SlopeEst): #needs correction
                corr = posErr/self.SlopeEst
                
                corr = round(0.8*corr/.05)*.05
                
                print corr
                
                self.LastPos = currPos
                self.piezo.MoveTo(0,self.piezo.GetPos(0) + corr)
                
                self.LastStep = corr
                
    
    def DoSlopeEst(self):
        curPzPos = self.piezo.GetPos(0)
        curDetPos = self.posFcn()
        
        self.piezo.MoveTo(0,curPzPos - 1.0)
        time.sleep(10)
        #self.webcam.Refresh()
        detPosMinus = self.posFcn()
        
        self.piezo.MoveTo(0,curPzPos + 1.0)
        time.sleep(10)
        #self.webcam.Refresh()
        
        detPosPlus = self.posFcn()
        
        self.SlopeEst = (detPosMinus - detPosPlus)/2
        
        print 'Slope estimated at: %f pixels/um' % self.SlopeEst
        
        self.piezo.MoveTo(0,curPzPos)

        
        
                
                
                
            
        
