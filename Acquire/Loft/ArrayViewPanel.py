#!/usr/bin/python

##################
# ArrayViewPanel.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:FramePanel:ArrayViewPanel

import wx
import scipy

[wxID_ARRAYVIEWPANEL, wxID_ARRAYVIEWPANELDATAWINDOW, 
] = [wx.NewId() for _init_ctrls in range(2)]

class ArrayViewPanel(wx.Panel):
    def _init_coll_boxSizer1_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.DataWindow, 0, border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizer1 = wx.BoxSizer(orient=wx.VERTICAL)

        self._init_coll_boxSizer1_Items(self.boxSizer1)

        self.SetSizer(self.boxSizer1)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_ARRAYVIEWPANEL, name='ArrayViewPanel',
              parent=prnt, pos=wx.Point(289, 276), size=wx.Size(496, 323),
              style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(488, 296))

        self.DataWindow = wx.ScrolledWindow(id=wxID_ARRAYVIEWPANELDATAWINDOW,
              name='DataWindow', parent=self, pos=wx.Point(0, 0),
              size=wx.Size(488, 296), style=wx.HSCROLL | wx.VSCROLL)
        self.DataWindow.Bind(wx.EVT_PAINT, self.OnDataWindowPaint)
        self.DataWindow.Bind(wx.EVT_KEY_DOWN, self.OnDataWindowKeyDown)

        self._init_sizers()

    def __init__(self, parent, id, pos, size, style, name):
        self._init_ctrls(parent)
        self.data = None
        
        #self.XY = 'XY'
        #self.XZ = 'XZ'
        #self.YZ = 'YZ'
        
        self.View = self.SliceXY
        self.Render = self.RenderGrey
        
    def SetData(self, data):
        self.data = data
        self.pos = scipy.zeros(len(data.shape))
        self.clim = (data.min(), data.max())
        
    def SetPos(self, pos):
        self.pos = pos
        
    def SetView(self, view):
        self.view = view
        
    
    def SliceXY(self, pos):
        if (len(pos) == 2):
            return self.data
        elif (len(pos) == 3):
            return self.data[:,:,pos[2]]
        elif (len(pos) == 4):
            return self.data[:,:,pos[2],pos[3]]
        
    def SliceXZ(self, pos):
        if (len(pos) == 2):
            print "Data has no z-dimension! - doing nothing"
        elif (len(pos) == 3):
            return self.data[:,pos[1], :]
        elif (len(pos) == 4):
            return self.data[:,pos[1],:,pos[3]]
        
    def SliceYZ(self, pos):
        if (len(pos) == 2):
            print "Data has no z-dimension! - doing nothing"
        elif (len(pos) == 3):
            return self.data[pos[0],:, :]
        elif (len(pos) == 4):
            return self.data[pos[0],:,:,pos[3]]
        
    def RenderGrey(self):
        dSlice = self.View(self.pos).copy()
        
        dSlice = dSlice - self.clim[0]
        dSlice = (255*dSlice/(self.clim[1]- self.clim[0])).astype('uint8')
        dSlice = dSlice.reshape((dSlice.shape[0], dSlice.shape[1],1))
        
        return wx.BitmapFromBuffer(dSlice.shape[0], dSlice.shape[1], scipy.concatenate((dSlice, dSlice, dSlice),2).ravel())

    def OnDataWindowPaint(self, event):
        if not (self.data == None):
            bm = self.Render()
            #self.imagepanel.SetVirtualSize(bm.GetSize())
            
            dc = wx.AutoBufferedPaintDC(self.DataWindow)
            dc.SetUserScale(5,5)
            self.DataWindow.PrepareDC(dc)
            dc.BeginDrawing()
            dc.DrawBitmap(bm, 0,0)
            dc.EndDrawing()
        else:
            event.Skip()

    def OnDataWindowKeyDown(self, event):
        if event.GetKeyCode() == wx.WXK_PRIOR:
            if (self.pos[2] > 0):
                self.pos[2] -=1
            
            self.DataWindow.Refresh()
        elif event.GetKeyCode() == wx.WXK_NEXT:
            if (self.pos[2] < (self.data.shape[2] - 1)):
                self.pos[2] +=1
            
            self.DataWindow.Refresh()
        else:
            event.Skip()
        #event.Skip()
        
        
