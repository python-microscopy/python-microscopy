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

class Splitter:
    def __init__(self, parent, menu, scope, dir='up_down', flipChan=1):
        self.dir = dir
        self.scope = scope
        self.flipChan=flipChan

        scope.splitting='none'

        self.constrainROI = False
        self.flipView = False

        idConstROI = wx.NewId()
        idFlipView = wx.NewId()

        self.menu = wx.Menu(title = '')

        self.menu.AppendCheckItem(idConstROI, 'Constrain ROI')
        wx.EVT_MENU(parent, idConstROI, self.OnConstrainROI)

        self.menu.AppendCheckItem(idFlipView, 'Flip view')
        wx.EVT_MENU(parent, idFlipView, self.OnFlipView)

        menu.AppendSeparator()
        menu.AppendMenu(-1, '&Splitter', self.menu)
        

    def OnConstrainROI(self,event):
        self.constrainROI = not self.constrainROI
        if self.constrainROI:
            self.scope.splitting = self.dir
        else:
            self.scope.splitting = 'none'

    def OnFlipView(self,event):
        self.flipView = not self.flipView
        if self.flipView:
            self.scope.vp.do.setFlip(self.flipChan, 1)
        else:
            self.scope.vp.do.setFlip(self.flipChan, 0)

    def Unmix(self, mixingMatrix, offset):
        import scipy.linalg
        from PYME import cSMI
        from pylab import *

        umm = scipy.linalg.inv(mixingMatrix)

        dsa = cSMI.CDataStack_AsArray(self.scope.pa.ds, 0) - offset

        g_ = dsa[:, :(dsa.shape[1]/2)]
        r_ = dsa[:, (dsa.shape[1]/2):]
        r_ = fliplr(r_)

        g = umm[0,0]*g_ + umm[0,1]*r_
        r = umm[1,0]*g_ + umm[1,1]*r_

        figure()
        subplot(211)
        imshow(g, cmap=cm.hot)

        subplot(212)
        imshow(r, cmap=cm.hot)



   
    

        

        
