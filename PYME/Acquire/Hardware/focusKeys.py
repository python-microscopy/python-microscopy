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
    def __init__(self, parent, piezo, keys = ['F1', 'F2', 'F3', 'F4'], scope = None):
        self.piezo = piezo
        self.focusIncrement = 0.2
        self.scope = scope
        
        parent.AddMenuItem('Focus', 'Focus Down\t%s' % keys[0], self.OnFocDown)
        parent.AddMenuItem('Focus', 'Focus Up\t%s' % keys[1], self.OnFocUp)
        parent.AddMenuItem('Focus', 'Sensitivity Down\t%s' % keys[2], self.OnSensDown)
        parent.AddMenuItem('Focus', 'Sensitivity Up\t%s' % keys[3], self.OnSensUp)

#        idFocUp = wx.NewIdRef()
#        idFocDown = wx.NewIdRef()
#        idSensUp = wx.NewIdRef()
#        idSensDown = wx.NewIdRef()

#        self.menu = wx.Menu(title = '')
#
#        self.menu.Append(idFocDown, 'Focus Down\t%s' % keys[0])
#        wx.EVT_MENU(parent, idFocDown, self.OnFocDown)
#
#        self.menu.Append(idFocUp, 'Focus Up\t%s' % keys[1])
#        wx.EVT_MENU(parent, idFocUp, self.OnFocUp)
#
#        self.menu.Append(idSensDown, 'Sensitivity Down\t%s' % keys[2])
#        wx.EVT_MENU(parent, idSensDown, self.OnSensDown)
#
#        self.menu.Append(idSensUp, 'Sensitivity Up\t%s' % keys[3])
#        wx.EVT_MENU(parent, idSensUp, self.OnSensUp)
#
#        menu.Append(menu=self.menu, title = 'Focus')
        self.mbar = parent.menubar
        self.mpos = self.mbar.GetMenuCount() - 1


    def OnFocDown(self,event):
        #p = self.piezo[0].GetPos(self.piezo[1])
        if self.scope and 'zs' in dir(self.scope) and self.scope.zs.running:
            #special case for when we are using a wavetable - move whole stack
            self.scope.frameWrangler.stop()
            self.scope.zs.zPoss -= self.focusIncrement
            self.scope.frameWrangler.start()
        else:
            try:
                self.piezo[0].MoveRel(self.piezo[1], -self.focusIncrement, False)
            except AttributeError:
                if 'lastPos' in dir(self.piezo[0]):
                    p = self.piezo[0].lastPos
                else:
                    p = self.piezo[0].GetPos(self.piezo[1])

                self.piezo[0].MoveTo(self.piezo[1], p - self.focusIncrement, False)

    def OnFocUp(self,event):
        if self.scope and 'zs' in dir(self.scope) and self.scope.zs.running:
            #special case for when we are using a wavetable - move whole stack
            self.scope.frameWrangler.stop()
            self.scope.zs.zPoss += self.focusIncrement
            self.scope.frameWrangler.start()
        else:
            try:
                self.piezo[0].MoveRel(self.piezo[1], self.focusIncrement, False)
            except AttributeError:
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
            
        self.mbar.SetMenuLabel(self.mpos, 'Focus = %3.2f, Inc = %3.2fum' %(p, self.focusIncrement))

class PositionKeys:
    def __init__(self, parent, xpiezo, ypiezo, keys = ['F9', 'F10', 'F11', 'F12'], scope = None):
        self.xpiezo = xpiezo
        self.ypiezo = ypiezo
        
        self.focusIncrement = 0.03
        self.scope = scope

        parent.AddMenuItem('Position', 'Position Down\t%s' % keys[0], self.OnDown)
        parent.AddMenuItem('Position', 'Position Up\t%s' % keys[1], self.OnUp)
        parent.AddMenuItem('Position', 'Position Left\t%s' % keys[2], self.OnLeft)
        parent.AddMenuItem('Position', 'Position Right\t%s' % keys[3], self.OnRight)
        parent.AddMenuItem('Position', 'Sensitivity Down\tCtrl-N', self.OnSensDown)
        parent.AddMenuItem('Position', 'Sensitivity Up\tCtrl-M', self.OnSensUp)

        # idFocUp = wx.NewIdRef()
        # idFocDown = wx.NewIdRef()
        # idSensUp = wx.NewIdRef()
        # idSensDown = wx.NewIdRef()

        # self.menu = wx.Menu(title = '')

        # self.menu.Append(idFocDown, 'Left\t%s' % keys[0])
        # wx.EVT_MENU(parent, idFocDown, self.OnLeft)

        # self.menu.Append(idFocUp, 'Right\t%s' % keys[1])
        # wx.EVT_MENU(parent, idFocUp, self.OnRight)

        # self.menu.Append(idSensDown, 'Up\t%s' % keys[2])
        # wx.EVT_MENU(parent, idSensDown, self.OnUp)

        # self.menu.Append(idSensUp, 'Down\t%s' % keys[3])
        # wx.EVT_MENU(parent, idSensUp, self.OnDown)

        # menu.Append(menu=self.menu, title = 'Position')
        self.mbar = parent.menubar
        self.mpos = self.mbar.GetMenuCount() - 1


    def OnLeft(self,event):
        if 'lastPos' in dir(self.xpiezo[0]):
            p = self.xpiezo[0].lastPos[self.xpiezo[1]-1]
        else:
            p = self.xpiezo[0].GetPos(self.xpiezo[1])
            
        self.xpiezo[0].MoveTo(self.xpiezo[1], p - self.focusIncrement, False, vel=10)

    def OnRight(self,event):
        if 'lastPos' in dir(self.xpiezo[0]):
            p = self.xpiezo[0].lastPos[self.xpiezo[1]-1]
        else:
            p = self.xpiezo[0].GetPos(self.xpiezo[1])
            
        self.xpiezo[0].MoveTo(self.xpiezo[1], p + self.focusIncrement, False, vel=10)

    def OnUp(self,event):
        if 'lastPos' in dir(self.ypiezo[0]):
            p = self.ypiezo[0].lastPos[self.ypiezo[1]-1]
        else:
            p = self.ypiezo[0].GetPos(self.ypiezo[1])
            
        self.ypiezo[0].MoveTo(self.ypiezo[1], p - self.focusIncrement, False, vel=10)
        
    def OnDown(self,event):
        if 'lastPos' in dir(self.ypiezo[0]):
            p = self.ypiezo[0].lastPos[self.ypiezo[1]-1]
        else:
            p = self.ypiezo[0].GetPos(self.ypiezo[1])
            
        self.ypiezo[0].MoveTo(self.ypiezo[1], p + self.focusIncrement, False, vel=10)

    def OnSensDown(self,event):
        if self.focusIncrement > 0.001:
            self.focusIncrement /= 2.

    def OnSensUp(self,event):
        if self.focusIncrement < 0.04:
            self.focusIncrement *= 2.

    def refresh(self):            
        self.mbar.SetMenuLabel(self.mpos, 'Position Inc =  %3.2fum' %(1000*self.focusIncrement))

#    def refresh(self):
#        if 'lastPos' in dir(self.piezo[0]):
#            p = self.piezo[0].lastPos
#        else:
#            p = self.piezo[0].GetPos(self.piezo[1])
#            
#        self.mbar.SetMenuLabel(self.mpos, 'Focus = %3.2f, Inc = %3.2f' %(p, self.focusIncrement))

        

        
