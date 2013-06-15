#!/usr/bin/python

##################
# focusKeys.py
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

class FocusKeys:
    def __init__(self, parent, menu, piezo, keys = ['F1', 'F2', 'F3', 'F4'], scope = None):
        self.piezo = piezo
        self.focusIncrement = 0.2
        self.scope = scope

        idFocUp = wx.NewId()
        idFocDown = wx.NewId()
        idSensUp = wx.NewId()
        idSensDown = wx.NewId()

        self.menu = wx.Menu(title = '')

        self.menu.Append(idFocDown, 'Focus Down\t%s' % keys[0])
        wx.EVT_MENU(parent, idFocDown, self.OnFocDown)

        self.menu.Append(idFocUp, 'Focus Up\t%s' % keys[1])
        wx.EVT_MENU(parent, idFocUp, self.OnFocUp)

        self.menu.Append(idSensDown, 'Sensitivity Down\t%s' % keys[2])
        wx.EVT_MENU(parent, idSensDown, self.OnSensDown)

        self.menu.Append(idSensUp, 'Sensitivity Up\t%s' % keys[3])
        wx.EVT_MENU(parent, idSensUp, self.OnSensUp)

        menu.Append(menu=self.menu, title = 'Focus')
        self.mbar = menu
        self.mpos = menu.GetMenuCount() - 1


    def OnFocDown(self,event):
        #p = self.piezo[0].GetPos(self.piezo[1])
        if self.scope and 'zs' in dir(self.scope) and self.scope.zs.running:
            #special case for when we are using a wavetable - move whole stack
            self.scope.pa.stop()
            self.scope.zs.zPoss -= self.focusIncrement
            self.scope.pa.start()
        else:
            if 'lastPos' in dir(self.piezo[0]):
                p = self.piezo[0].lastPos
            else:
                p = self.piezo[0].GetPos(self.piezo[1])
                
            self.piezo[0].MoveTo(self.piezo[1], p - self.focusIncrement, False)

    def OnFocUp(self,event):
        if self.scope and 'zs' in dir(self.scope) and self.scope.zs.running:
            #special case for when we are using a wavetable - move whole stack
            self.scope.pa.stop()
            self.scope.zs.zPoss += self.focusIncrement
            self.scope.pa.start()
        else:
            if 'lastPos' in dir(self.piezo[0]):
                p = self.piezo[0].lastPos
            else:
                p = self.piezo[0].GetPos(self.piezo[1])
                
            self.piezo[0].MoveTo(self.piezo[1], p + self.focusIncrement, False)

    def OnSensDown(self,event):
        if self.focusIncrement > 0.05:
            self.focusIncrement /= 2.

    def OnSensUp(self,event):
        if self.focusIncrement < 10:
            self.focusIncrement *= 2.

    def refresh(self):
        if 'lastPos' in dir(self.piezo[0]):
            p = self.piezo[0].lastPos
        else:
            p = self.piezo[0].GetPos(self.piezo[1])
            
        self.mbar.SetMenuLabel(self.mpos, 'Focus = %3.2f, Inc = %3.2f' %(p, self.focusIncrement))

class PositionKeys:
    def __init__(self, parent, menu, xpiezo, ypiezo, keys = ['F9', 'F10', 'F11', 'F12'], scope = None):
        self.xpiezo = xpiezo
        self.ypiezo = ypiezo
        
        self.focusIncrement = 0.03
        self.scope = scope

        idFocUp = wx.NewId()
        idFocDown = wx.NewId()
        idSensUp = wx.NewId()
        idSensDown = wx.NewId()

        self.menu = wx.Menu(title = '')

        self.menu.Append(idFocDown, 'Left\t%s' % keys[0])
        wx.EVT_MENU(parent, idFocDown, self.OnLeft)

        self.menu.Append(idFocUp, 'Right\t%s' % keys[1])
        wx.EVT_MENU(parent, idFocUp, self.OnRight)

        self.menu.Append(idSensDown, 'Up\t%s' % keys[2])
        wx.EVT_MENU(parent, idSensDown, self.OnUp)

        self.menu.Append(idSensUp, 'Down\t%s' % keys[3])
        wx.EVT_MENU(parent, idSensUp, self.OnDown)

        menu.Append(menu=self.menu, title = 'Position')
        self.mbar = menu
        self.mpos = menu.GetMenuCount() - 1


    def OnLeft(self,event):
        if 'lastPos' in dir(self.xpiezo[0]):
            p = self.xpiezo[0].lastPos[self.xpiezo[1]-1]
        else:
            p = self.xpiezo[0].GetPos(self.xpiezo[1])
            
        self.xpiezo[0].MoveTo(self.xpiezo[1], p - self.focusIncrement, False)

    def OnRight(self,event):
        if 'lastPos' in dir(self.xpiezo[0]):
            p = self.xpiezo[0].lastPos[self.xpiezo[1]-1]
        else:
            p = self.xpiezo[0].GetPos(self.xpiezo[1])
            
        self.xpiezo[0].MoveTo(self.xpiezo[1], p + self.focusIncrement, False)

    def OnUp(self,event):
        if 'lastPos' in dir(self.ypiezo[0]):
            p = self.ypiezo[0].lastPos[self.ypiezo[1]-1]
        else:
            p = self.ypiezo[0].GetPos(self.ypiezo[1])
            
        self.ypiezo[0].MoveTo(self.ypiezo[1], p - self.focusIncrement, False)
        
    def OnDown(self,event):
        if 'lastPos' in dir(self.ypiezo[0]):
            p = self.ypiezo[0].lastPos[self.ypiezo[1]-1]
        else:
            p = self.ypiezo[0].GetPos(self.ypiezo[1])
            
        self.ypiezo[0].MoveTo(self.ypiezo[1], p + self.focusIncrement, False)

    
#    def refresh(self):
#        if 'lastPos' in dir(self.piezo[0]):
#            p = self.piezo[0].lastPos
#        else:
#            p = self.piezo[0].GetPos(self.piezo[1])
#            
#        self.mbar.SetMenuLabel(self.mpos, 'Focus = %3.2f, Inc = %3.2f' %(p, self.focusIncrement))

        

        
