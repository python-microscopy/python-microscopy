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
from PYME.DSView.myviewpanel_numarray import MyViewPanel
import numpy

class Splitter:
    def __init__(self, parent, menu, scope, dir='up_down', flipChan=1):
        self.dir = dir
        self.scope = scope
        self.flipChan=flipChan
        self.parent = parent

        scope.splitting='none'

        self.offset = 0
        self.mixMatrix = numpy.array([[1.,0.],[0.,1.]])

        self.constrainROI = False
        self.flipView = False

        idConstROI = wx.NewId()
        idFlipView = wx.NewId()
        idUnmix = wx.NewId()
        idShiftfield = wx.NewId()

        self.menu = wx.Menu(title = '')

        self.menu.AppendCheckItem(idConstROI, 'Constrain ROI')
        wx.EVT_MENU(parent, idConstROI, self.OnConstrainROI)

        self.menu.AppendCheckItem(idFlipView, 'Flip view')
        wx.EVT_MENU(parent, idFlipView, self.OnFlipView)
        self.menu.Append(idUnmix, 'Unmix\tF7')
        wx.EVT_MENU(parent, idUnmix, self.OnUnmix)

        self.menu.Append(idShiftfield, 'Set Shift Field')
        wx.EVT_MENU(parent, idShiftfield, self.OnSetShiftField)

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

    def OnUnmix(self,event):
        #self.Unmix()
        f = UnMixFrame(self.parent, splitter = self)
        f.SetSize((800,500))
        f.Show()

    def OnSetShiftField(self, event):
        fdialog = wx.FileDialog(None, 'Select shift field',
            wildcard='*.sf', style=wx.OPEN)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            self.SetShiftField(numpy.load(fdialog.GetPath().encode()))

    def SetShiftField(self, shiftField):
        #self.shiftField = shiftField
        X, Y = numpy.ogrid[:512, :256]
        self.X2 = numpy.round(X - shiftField[0](X*70., Y*70.)/70.).astype('i')
        self.Y2 = numpy.round(Y - shiftField[1](X*70., Y*70.)/70.).astype('i')

    def _deshift(self, red_chan):
        if 'X2' in dir(self):
            x1 = self.scope.cam.GetROIX1() - 1
            x2 = self.scope.cam.GetROIX2()
            y1 = self.scope.cam.GetROIY1() - 1

            print self.X2.shape

            Xn = self.X2[x1:x2, y1:(y1 + red_chan.shape[1])] - x1
            Yn = self.Y2[x1:x2, y1:(y1 + red_chan.shape[1])] - y1

            print Xn.shape

            Xn = numpy.maximum(numpy.minimum(Xn, red_chan.shape[0]-1), 0)
            Yn = numpy.maximum(numpy.minimum(Yn, red_chan.shape[1]-1), 0)

            return red_chan[Xn, Yn]

        else:
            return red_chan


    def Unmix(self):
        import scipy.linalg
        from PYME import cSMI
        from pylab import *
        from PYME.DSView.dsviewer_npy import View3D

        umm = scipy.linalg.inv(self.mixMatrix)

        dsa = cSMI.CDataStack_AsArray(self.scope.pa.ds, 0).squeeze() - self.offset

        g_ = dsa[:, :(dsa.shape[1]/2)]
        r_ = dsa[:, (dsa.shape[1]/2):]
        r_ = self._deshift(fliplr(r_))

        print g_.shape, r_.shape

        g = umm[0,0]*g_ + umm[0,1]*r_
        r = umm[1,0]*g_ + umm[1,1]*r_

        g = g*(g > 0)
        r = r*(r > 0)

#        figure()
#        subplot(211)
#        imshow(g.T, cmap=cm.hot)
#
#        subplot(212)
#        imshow(r.T, cmap=cm.hot)

        #View3D([r.reshape(r.shape + (1,)),g.reshape(r.shape + (1,))])
        return [r.reshape(r.shape + (1,)),g.reshape(r.shape + (1,))]


class UnMixFrame(wx.Frame):
    def __init__(self, parent=None, title='Unmixing', splitter = None, size=(800, 500)):
        wx.Frame.__init__(self,parent, -1, title, size=size)

        self.splitter = splitter

        self.ds = self.splitter.Unmix()
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        pan = wx.Panel(self, -1)
        psizer = wx.BoxSizer(wx.VERTICAL)

        bsizer = wx.StaticBoxSizer(wx.StaticBox(pan, -1, 'Mix Matrix'), wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tMM00 = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.mixMatrix[0,0]), size=(40,-1))
        hsizer.Add(self.tMM00, 1, wx.ALL,2 )

        self.tMM01 = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.mixMatrix[0,1]), size=(40,-1))
        hsizer.Add(self.tMM01, 1, wx.ALL,2 )

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tMM10 = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.mixMatrix[1,0]), size=(40,-1))
        hsizer.Add(self.tMM10, 1, wx.ALL,2 )

        self.tMM11 = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.mixMatrix[1,1]), size=(40,-1))
        hsizer.Add(self.tMM11, 1, wx.ALL,2 )

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        psizer.Add(bsizer, 0, wx.ALL, 5)


        bsizer = wx.StaticBoxSizer(wx.StaticBox(pan, -1, 'Offset'), wx.HORIZONTAL)
        self.tOffset = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.offset), size=(40,-1))

        bsizer.Add(self.tOffset, 1, wx.ALL, 0)
        psizer.Add(bsizer, 0, wx.ALL|wx.EXPAND, 5)

#        self.bUpdate = wx.Button(pan, -1, 'Update')
#        psizer.Add(self.bUpdate, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)
#        self.bUpdate.Bind(wx.EVT_BUTTON, self.OnUpdateMix)

        pan.SetSizerAndFit(psizer)

        sizer.Add(pan, 0, 0, 0)

        self.vp = MyViewPanel(self, self.ds)
        sizer.Add(self.vp, 1,wx.EXPAND,0)
        self.SetAutoLayout(1)
        self.SetSizerAndFit(sizer)

        #self.Layout()
        #self.update()
        wx.EVT_CLOSE(self, self.OnCloseWindow)
        #wx.EVT_SIZE(self, self.OnSize)

        #self.statusbar = self.CreateStatusBar(1, wx.ST_SIZEGRIP)

        #self.Layout()
        
        self.splitter.scope.pa.WantFrameGroupNotification.append(self.update)

    def update(self, caller=None):
        #print self.tMM00.GetValue(), self.tMM01.GetValue()
        self.splitter.mixMatrix[0,0]= float(self.tMM00.GetValue())
        self.splitter.mixMatrix[0,1]= float(self.tMM01.GetValue())
        self.splitter.mixMatrix[1,0]= float(self.tMM10.GetValue())
        self.splitter.mixMatrix[1,1]= float(self.tMM11.GetValue())
        self.splitter.offset= float(self.tOffset.GetValue())

        self.vp.ResetDataStack(self.splitter.Unmix())
        self.vp.imagepanel.Refresh()

    def OnCloseWindow(self, event):
        self.splitter.scope.pa.WantFrameGroupNotification.remove(self.update)
        self.Destroy()

    def OnUpdateMix(self, event):
        self.splitter.mixMatrix[0,0]= float(self.tMM00.GetValue())
        self.splitter.mixMatrix[0,1]= float(self.tMM01.GetValue())
        self.splitter.mixMatrix[1,0]= float(self.tMM10.GetValue())
        self.splitter.mixMatrix[1,1]= float(self.tMM11.GetValue())
        self.splitter.offset= float(self.tOffset.GetValue())

        print self.splitter.mixMatrix

        self.update()

    def OnSize(self, event):
        self.Layout()
        event.Skip()


   
    

        

        
