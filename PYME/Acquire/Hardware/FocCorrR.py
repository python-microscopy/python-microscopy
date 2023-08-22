#!/usr/bin/python

##################
# FocCorrR.py
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
import Pyro.core
import eventLog

class FocusCorrector(wx.Timer):
    def __init__(self, piezo, tolerance=0.2, estSlopeDyn=False, recDrift=False, axis='X', guideLaser=None):
        wx.Timer.__init__(self)
        
        self.piezo = piezo
        self.COIi = Pyro.core.getProxyForURI('PYRONAME://COIi')
        
        self.tolerance = tolerance #set a position tolerance (default 200nm)
        
        self.Tracking = False #we're not locked at the start
        #self.SlopeEst = 1 #pretty arbitray

        self.TargetPos = 0

        self.cumShift=0
        
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
        if  not self.guideLaser is None:
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

            eventLog.logEvent('Focus Position', '%f, %f, %f' % (currPos, posErr,self.SlopeEst ))
            
            if abs(posErr) > abs(self.tolerance*self.SlopeEst): #needs correction
                corr = posErr/self.SlopeEst
                
                corr = round(0.8*corr/.05)*.05
                
                #don't make more than 1um correction in any one step
                corr = min(max(corr, -1), 1)
                print(corr)
                
                self.LastPos = currPos
                self.piezo.MoveTo(0,self.piezo.GetPos(0) + corr)
                
                self.LastStep = corr

                self.cumShift += corr #increment cumulative shift

                #we've moved a lot - lock obviously broken
                if abs(self.cumShift) > 5: 
                    self.Tracking = False #turn tracking off to stop runnaway
            else:
                self.cumShift = 0 #reset cumulative shift
                
    
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
        
        print(( 'Slope estimated at: %f pixels/um' % self.SlopeEst))
        
        self.piezo.MoveTo(0,curPzPos)

        
    def GetStatus(self):
        p = self.posFcn() - self.TargetPos
        if 'SlopeEst' in dir(self):
            stext = 'Focus: %3.2f[%3.3fum]' % (p, p/self.SlopeEst)
        else: #we haven't estimated the slope yet
            stext = 'Focus: %3.2f' % (p,)

        if self.Tracking:
            stext += ' [locked]'

        return stext

    def TOn(self, event=None):
        self.TrackOn(False)

    def TOnCalc(self, event=None):
        self.TrackOn(True)

    def TOff(self, event=None):
        self.TrackOff()

    def addMenuItems(self,parentWindow, menu):
        """Add menu items and keyboard accelerators for LED control
        to the specified menu & parent window"""
        #Create IDs
        self.ID_TRACK_ON = wx.NewIdRef()
        self.ID_TRACK_ON_CALC = wx.NewIdRef()
        self.ID_TRACK_OFF = wx.NewIdRef()
        
        mTracking = wx.Menu(title = '')

        #Add menu items
        mTracking.Append(self.ID_TRACK_ON,
              'Tracking ON', kind=wx.ITEM_NORMAL)
        
        mTracking.Append(self.ID_TRACK_ON_CALC,
              'Tracking ON (Recalc)', kind=wx.ITEM_NORMAL)
        mTracking.Append(self.ID_TRACK_OFF,
              'Tracking OFF', kind=wx.ITEM_NORMAL)


        menu.Append(mTracking, title = 'Autofocus')

        
        #Handle clicking on the menu items
        wx.EVT_MENU(parentWindow, self.ID_TRACK_ON, self.TOn)
        wx.EVT_MENU(parentWindow, self.ID_TRACK_ON_CALC, self.TOnCalc)
        wx.EVT_MENU(parentWindow, self.ID_TRACK_OFF, self.TOff)
