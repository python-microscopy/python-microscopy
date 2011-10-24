#!/usr/bin/python

##################
# focusKeys.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
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


        

        
