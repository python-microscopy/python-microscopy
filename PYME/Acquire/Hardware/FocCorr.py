#!/usr/bin/python

##################
# FocCorr.py
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

import wx
import time

class FocusCorrector(wx.Timer):
    def __init__(self, piezo, webcam, tolerance=0.2, estSlopeDyn=False):
        wx.Timer.__init__(self)
        
        self.piezo = piezo
        self.webcam = webcam
        
        self.tolerance = tolerance #set a position tolerance (default 200nm)
        
        self.Tracking = False #we're not locked at the start
        self.SlopeEst = 0.5 #pretty arbitray - 200nm/pixel - will be refined as algorithm runs
        self.TargetPos = None
        
        self.EstimateSlope = estSlopeDyn
        
    def TrackOn(self):
        self.TargetPos = self.webcam.COIX #just use the vertical deflection for now
        
        self.DoSlopeEst()
        
        #fudge a previous correction to make our automatic slope correction work
        self.LastPos = self.TargetPos - self.SlopeEst
        self.LastStep = 1.0
        
        self.Tracking = True
        
    def TrackOff(self):
        self.Tracking = False
        
    def Notify(self):
        if self.Tracking:
            currPos = self.webcam.COIX
            if self.EstimateSlope:
                SlopeEst_new = (self.LastPos - currPos)/self.LastStep #estimate slope
            
                self.SlopeEst = .5*self.SlopeEst + 0.5*SlopeEst_new #average slope changes
                print((self.SlopeEst))
            
            posErr = currPos - self.TargetPos
            print(posErr)
            
            if abs(posErr) > abs(self.tolerance/self.SlopeEst): #needs correction
                corr = posErr*self.SlopeEst
                
                corr = round(0.8*corr/.05)*.05
                
                print(corr)
                
                self.LastPos = currPos
                self.piezo.MoveTo(0,self.piezo.GetPos(0) + corr)
                
                self.LastStep = corr
                
    
    def DoSlopeEst(self):
        curPzPos = self.piezo.GetPos(0)
        curDetPos = self.webcam.COIX
        
        self.piezo.MoveTo(0,curPzPos - 1.0)
        time.sleep(1)
        self.webcam.Refresh()
        detPosMinus = self.webcam.COIX
        
        self.piezo.MoveTo(0,curPzPos + 1.0)
        time.sleep(1)
        self.webcam.Refresh()
        
        detPosPlus = self.webcam.COIX
        
        self.SlopeEst = (detPosMinus - detPosPlus)/2
        
        print(('Slope estimated at: %f pixels/um' % self.SlopeEst))
        
        self.piezo.MoveTo(0,curPzPos)
        
                
                
                
            
        