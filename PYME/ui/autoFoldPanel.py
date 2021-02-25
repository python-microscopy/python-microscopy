#!/usr/bin/python

###############
# autoFoldPanel.py
#
# Copyright David Baddeley, 2012
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
################
import wx
#from wx.lib.agw import aui
import math
import six
from wx.lib.agw.aui.aui_utilities import BitmapFromBits

def ColourFromStyle(col):
    if isinstance(col, six.string_types):
        col = wx.NamedColour(col)
    else:
        col = wx.Colour(*col)

    return col

class SizeReportCtrl(wx.PyControl):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition,
                size=(300, 150), mgr=None):

        wx.PyControl.__init__(self, parent, id, pos, size, style=wx.NO_BORDER)
        self._mgr = mgr

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)

        #self.SetSize(-1, 300)
        #self.SetMinSize((300, 150))


    def OnPaint(self, event):
    
        dc = wx.PaintDC(self)
        size = self.GetClientSize()

        s = "Size: %d x %d"%(size.x, size.y)

        dc.SetFont(wx.NORMAL_FONT)
        w, height = dc.GetTextExtent(s)
        height += 3
        dc.SetBrush(wx.WHITE_BRUSH)
        dc.SetPen(wx.WHITE_PEN)
        dc.DrawRectangle(0, 0, size.x, size.y)
        dc.SetPen(wx.LIGHT_GREY_PEN)
        dc.DrawLine(0, 0, size.x, size.y)
        dc.DrawLine(0, size.y, size.x, 0)
        dc.DrawText(s, (size.x-w)/2, (size.y-height*5)/2)

        if self._mgr:
        
            pi = self._mgr.GetPane(self)

            s = "Layer: %d"%pi.dock_layer
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*1))

            s = "Dock: %d Row: %d"%(pi.dock_direction, pi.dock_row)
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*2))

            s = "Position: %d"%pi.dock_pos
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*3))

            s = "Proportion: %d"%pi.dock_proportion
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*4))

        
    def OnEraseBackground(self, event):

        pass
    

    def OnSize(self, event):
    
        self.Refresh()
        self.Update()
        

#stolen from aui
pin_bits     = b'\xff\xff\xff\xff\xff\xff\x1f\xfc\xdf\xfc\xdf\xfc\xdf\xfc\xdf\xfc' \
               b'\xdf\xfc\x0f\xf8\x7f\xff\x7f\xff\x7f\xff\xff\xff\xff\xff\xff\xff'
""" Pin button bitmap for a pane. """




DEFAULT_CAPTION_STYLE = {
'HEIGHT'              : 20,
'FONT_COLOUR'         : 'BLACK',
#'FONT_WEIGHT' : wx.BOLD,
#'FONT_SIZE'           : 12,
'CAPTION_INDENT'      : 5,
'BACKGROUND_COLOUR_1' : (198, 198, 198), #default AUI caption colours
'BACKGROUND_COLOUR_2' : (226, 226, 226),
'INACTIVE_PIN_COLOUR' : (170, 170, 170),
'ACTIVE_PIN_COLOUR'   : (0, 0, 0),
'ELLIPSES_COLOUR'     : (170, 170, 170),
'ELLIPSES_RADIUS'     : 2,
}

class CaptionBar(wx.Window):

    def __init__(self, parent, id = wx.ID_ANY, pos=(-1,-1), caption="",
                 foldIcons=None, cbstyle=DEFAULT_CAPTION_STYLE):

        wx.Window.__init__(self, parent, id, pos=pos,
                           size=(-1,cbstyle['HEIGHT']), style=wx.NO_BORDER)

        self.style = cbstyle
        self.parent = parent
        self.caption = caption
        self.foldIcons = foldIcons

        self._inactive_pin_bitmap = BitmapFromBits(pin_bits, 16, 16, ColourFromStyle(self.style['INACTIVE_PIN_COLOUR']))
        self._active_pin_bitmap = BitmapFromBits(pin_bits, 16, 16, ColourFromStyle(self.style['ACTIVE_PIN_COLOUR']))

        self.pinButtonRect = (0,0,0,0)

#        if foldIcons is None:
#            foldIcons = wx.ImageList(16, 16)
#
#            bmp = ExpandedIcon.GetBitmap()
#            foldIcons.Add(bmp)
#            bmp = CollapsedIcon.GetBitmap()
#            foldIcons.Add(bmp)
#
#        # set initial size
#        if foldIcons:
#            assert foldIcons.GetImageCount() > 1
#            iconWidth, iconHeight = foldIcons.GetSize(0)


        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftClick)
        self.Bind(wx.EVT_ENTER_WINDOW, self.OnMouseEnterLeave)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseEnterLeave)

#        self.Bind(wx.EVT_CHAR, self.OnChar)


    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        gc = wx.GraphicsContext.Create(dc)

        gc.PushState()

        wndRect = self.GetRect()
#        vertical = self.IsVertical()

        #self.FillCaptionBackground(dc)
        #barHeight = self.style['HEIGHT']

        #gc.DrawRectangle(0,0, self.style['HEIGHT'],50,50,5)
        col_1 = ColourFromStyle(self.style['BACKGROUND_COLOUR_1'])
        col_2 = ColourFromStyle(self.style['BACKGROUND_COLOUR_2'])

        brush = gc.CreateLinearGradientBrush(0,0,0,self.style['HEIGHT'], col_1, col_2)
        gc.SetBrush(brush)

        gc.DrawRectangle(*wndRect)


        font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
        #print font.GetFaceName(), font.GetPointSize()
        if 'FONT_WEIGHT' in self.style.keys():
            font.SetWeight(self.style['FONT_WEIGHT'])
        if 'FONT_SIZE' in self.style.keys():
            font.SetPointSize(self.style['FONT_SIZE'])
            
        fc = ColourFromStyle(self.style['FONT_COLOUR'])
        gc.SetFont(font, fc)

        w,h = gc.GetTextExtent(self.caption)

        y0 = self.style['HEIGHT']/2. - h/2.
        gc.DrawText(self.caption, self.style['CAPTION_INDENT'], y0)


        h = self._active_pin_bitmap.GetHeight()
        w = self._active_pin_bitmap.GetWidth()

        y0 = self.style['HEIGHT']/2. - h/2.

        #print wndRect[2]

        self.pinButtonRect = (wndRect[2] - h - y0, y0, w,h)

        if self.parent.foldable:
            if self.parent.pinnedOpen:
                gc.DrawBitmap(self._active_pin_bitmap, *self.pinButtonRect)
            else:
                gc.DrawBitmap(self._inactive_pin_bitmap, *self.pinButtonRect)

        if self.parent.folded and self.parent.foldable:
            self.DrawEllipses(gc)

        gc.PopState()

    def DrawEllipses(self, gc):
        gc.SetBrush(wx.Brush(ColourFromStyle(self.style['ELLIPSES_COLOUR'])))
        path = gc.CreatePath()
        path.AddCircle(0, 0, self.style['ELLIPSES_RADIUS'])
        path.AddCircle(3*self.style['ELLIPSES_RADIUS'], 0, self.style['ELLIPSES_RADIUS'])
        path.AddCircle(6*self.style['ELLIPSES_RADIUS'], 0, self.style['ELLIPSES_RADIUS'])

        gc.PushState()
        gc.Translate(self.pinButtonRect[0], self.pinButtonRect[1] + self.pinButtonRect[3])
        bx = path.GetBox()
        gc.Translate(-bx[2], -bx[3])

        gc.FillPath(path)

        gc.PopState()



    def OnLeftClick(self, event):
        if wx.Rect(*self.pinButtonRect).Inside(event.GetPosition()):
            self.parent.TogglePin()


    def OnSize(self, event):
        rect = self.GetRect()
        self.RefreshRect(rect)

    def OnMouseEnterLeave(self, event):
        #event.ResumePropagation(2)
        #print event.ShouldPropagate()
        #print 'mev'
        event.SetEventObject(self.parent)
        event.ResumePropagation(2)
        event.Skip()
        








class foldElement:
    def __init__(self, window, foldable=True, foldedWindow=None):
        self.window = window
        self.foldable = foldable
        #if foldedWindow:
        self.foldedWindow = foldedWindow
#        elif self.foldable:
#            self.foldedWindow = wx.StaticText(self.window.GetParent(), -1, '...')
#            self.foldedWindow.Hide()

class foldingPane(wx.Panel):
    def __init__(self, *args, **kwargs):
        try:
            self.orientation = kwargs.pop('orientation')
        except KeyError:
            self.orientation = wx.VERTICAL

        try:
            self.padding = kwargs.pop('padding')
        except KeyError:
            self.padding = 5

        try:
            self.caption = kwargs.pop('caption')
        except KeyError:
            self.caption = None

        try:
            self.folded = kwargs.pop('folded')
        except KeyError:
            self.folded = True

        try:
            self.pinnedOpen = kwargs.pop('pinned')
            if self.pinnedOpen:
                self.folded = False
        except KeyError:
            self.pinnedOpen = False

        
        wx.Panel.__init__(self, * args, ** kwargs)

        if self.orientation == wx.VERTICAL:
            self.sizerflags = wx.EXPAND | wx.ALL
        else:
            self.sizerflags = wx.EXPAND | wx.RIGHT

        self.elements = []

        #without any elements, folding isn't going to do anything
        self.foldable = False

        self.sizer = wx.BoxSizer(self.orientation)

        if self.caption:
            #self.stCaption = wx.StaticText(self, -1, self.caption)
            self.stCaption = CaptionBar(self, -1, caption=self.caption)
            #self.stCaption.SetBackgroundColour(wx.TheColourDatabase.FindColour('SLATE BLUE'))
            self.sizer.Add(self.stCaption, 0, wx.EXPAND, 0)

        self.SetSizer(self.sizer)

    def AddNewElement(self, window, foldable=True, foldedWindow=None):
        self.AddElement(foldElement(window, foldable, foldedWindow))

    def AddElement(self, element):
        self.elements.append(element)
        
        self.sizer.Add(element.window, 0, self.sizerflags, self.padding)
        if element.foldable:
            self.foldable = True #we have at least one foldable element
            if element.foldedWindow:
                self.sizer.Add(element.foldedWindow, 0, self.sizerflags, self.padding)

            if self.folded:
                element.window.Hide()
                if element.foldedWindow:
                    element.foldedWindow.Show()
            else:
                element.window.Show()
                if element.foldedWindow:
                    element.foldedWindow.Hide()
                    
        #self.Fit()

    def PinOpen(self):
        self.Unfold()
        self.pinnedOpen = True
        self.stCaption.Refresh()

    def UnPin(self):
        self.pinnedOpen = False
        self.stCaption.Refresh()

    def TogglePin(self):
        if self.pinnedOpen:
            self.UnPin()
        else:
            self.PinOpen()

    def Fold(self):
        #print 'foo'
        if not self.folded and not self.pinnedOpen:
            for element in self.elements:
                #print self.folded, 'foo'
                if element.foldable:
                    element.window.Hide()
                    if element.foldedWindow:
                        element.foldedWindow.Show()

            self.folded = True
            self.stCaption.Refresh()
            #self.Layout()
            return True
        else:
            return False

            #self.Layout()

    def Unfold(self):
        #print 'bar'
        if self.folded:
            for element in self.elements:
                #print 'bar'
                if element.foldable:
                    element.window.Show()
                    #element.window.Layout()
                    if element.foldedWindow:
                        element.foldedWindow.Hide()

            self.folded = False
            self.stCaption.Refresh()
            #self.Layout()
            return True
        else:
            return False

            #self.Layout()


r_arrow = b'\xff\xff\xdf\xff\x9f\xff\x1f\xff\x5f\xfe\xdf\xfc\xdf\xf9\xdf\xf3\xdf' \
            b'\xf3\xdf\xf9\xdf\xfc\x5f\xfe\x1f\xff\x9f\xff\xdf\xff\xff\xff'

d_arrow = b'\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x80\xf3\xcf\xe7\xe7\xcf' \
            b'\xf3\x9f\xf9\x3f\xfc\x7f\xfe\xff\xff\xff\xff\xff\xff\xff\xff'


class foldButton(wx.Window):
    def __init__(self, parent, id=-1):
        wx.Window.__init__(self, parent, id, size=(16,16))

        self.bmR = BitmapFromBits(r_arrow, 16, 16, ColourFromStyle('BLACK'))
        self.bmD = BitmapFromBits(d_arrow, 16, 16, ColourFromStyle('BLACK'))

        self.folded = True

        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        if self.folded:
            dc.DrawBitmap(self.bmR, 0, 0, True)
        else:
            dc.DrawBitmap(self.bmD, 0, 0, True)

    def SetFolded(self, folded=True):
        self.folded = folded
        self.Refresh()
        self.Update()


class collapsingPane(foldingPane):
    def __init__(self, *args, **kwargs):
        try:
            caption = kwargs.pop('caption')
        except KeyError:
            caption = None

        #kwargs['folded'] = False

        foldingPane.__init__(self, *args, **kwargs)

        self.stCaption = wx.Panel(self, -1)

        self.bmR = BitmapFromBits(r_arrow, 16, 16, ColourFromStyle('BLACK'))
        self.bmD = BitmapFromBits(d_arrow, 16, 16, ColourFromStyle('BLACK'))

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #self.bFold = wx.BitmapButton(capPan, -1, bitmap=self.bmR, style = wx.NO_BORDER)
        self.bFold = foldButton(self.stCaption, -1)

        hsizer.Add(self.bFold, 0, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 2)
        hsizer.Add(wx.StaticText(self.stCaption, -1, caption), 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        self.stCaption.SetSizerAndFit(hsizer)
        self.sizer.Add(self.stCaption, 0, 0, 0)

        self.bFold.Bind(wx.EVT_LEFT_UP, self.OnFold)

    def OnFold(self, event):
        print('fold')
        if self.folded:
            self.Unfold()
            self.bFold.SetFolded(False)
        else:
            self.Fold()
            self.bFold.SetFolded(True)

        #self.Layout()
        #self.Fit()
        self.GetParent().GetParent().Layout()




from PYME.contrib import dispatch
class foldPanel(wx.Panel):
    def __init__(self, *args, **kwargs):
        try:
            self.orientation = kwargs.pop('orientation')
        except KeyError:
            self.orientation = wx.VERTICAL

        try:
            self.padding = kwargs.pop('padding')
        except KeyError:
            self.padding = 5

        wx.Panel.__init__(self, *args, **kwargs)

        if self.orientation == wx.VERTICAL:
            self.sizerflags = wx.EXPAND #| wx.BOTTOM
        else:
            self.sizerflags = wx.EXPAND #| wx.RIGHT

        self.priorities = []
        self.panes = []

        self.sizer = wx.BoxSizer(self.orientation)
        self.SetSizer(self.sizer)
        
        self.fold_signal = dispatch.Signal()

        #self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeave)

    def AddPane(self, window, priority=0):
        self.panes.append(window)
        self.priorities.append(priority)

        window.Bind(wx.EVT_ENTER_WINDOW, self.OnMouseEnterPane)
        window.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeavePane)

        self.RegenSizer()

    def RegenSizer(self):
        self.sizer.Clear()

        for pane, priority in zip(self.panes, self.priorities):
            self.sizer.Add(pane, priority, self.sizerflags, self.padding)

        self.sizer.AddStretchSpacer()

        self.sizer.Layout()

    def Clear(self):
        self.priorities = []
        for p in self.panes:
            p.Destroy()
        self.panes = []

        self.RegenSizer()
        
    def fold1(self):
        print('fold1')
        self.Layout()
        self.fold_signal.send(sender=self)
        self.Refresh()

    def OnMouseEnterPane(self, event):
        pane = event.GetEventObject()


#        ind = self.panes.index(pane)
#        print pane.GetMaxSize()
#        if self.orientation == wx.VERTICAL:
#            sc = float(pane.GetMaxSize()[1])/pane.GetSize()[1]
#        else:
#            sc = float(pane.GetMaxSize()[0])/pane.GetSize()[0]
#
#        print sc
#
#        self.priorities[ind] = math.ceil(sc)
        
        #if not pane.GetClientRect().Inside(event.GetPosition())
        #print 'enter', pane
        if pane.Unfold():
             #only re-do layout if we change state
             self.Layout()
             self.fold_signal.send(sender=self)
        #self.RegenSizer()

    def OnMouseLeavePane(self, event):
        pane = event.GetEventObject()
        #ind = self.panes.index(event.GetEventObject())
        #self.priorities[ind] = 1
        #print pane.GetClientRect()
        if wx.__version__ > '4':
            inside = pane.GetClientRect().Contains(event.GetPosition())
        else:
            inside = pane.GetClientRect().Inside(event.GetPosition())
            
        if not inside:
            if pane.Fold():
                self.Layout()
                self.fold_signal.send(sender=self)
        #event.Skip()
        

    def OnMouseLeave(self, event):
        self.Layout()
        self.fold_signal.send(sender=self)
        self.Refresh()
        #pass #self.RegenSizer()

if __name__ == "__main__":
    app = wx.PySimpleApp(0)
    wx.InitAllImageHandlers()
    f = wx.Frame(None, -1, "fpTest", size=(300, 800))
    p = foldPanel(f, -1)

    #for i in range(4):
    fi = foldingPane(p, -1, caption='Pane 1')
    sr = SizeReportCtrl(fi)
    sr.SetMinSize((300, 150))
    #sr.SetMaxSize((300, 300))
    fi.AddNewElement(sr)
    p.AddPane(fi)

    fi = foldingPane(p, -1, caption='pane 2')
    sr = SizeReportCtrl(fi)
    sr.SetMinSize((300, 150))
    #sr.SetMaxSize((300, 300))
    fi.AddNewElement(sr, foldable=False)
    p.AddPane(fi)

    fi = foldingPane(p, -1, caption='pane 3', pinned=True)
    sr = SizeReportCtrl(fi)
    sr.SetMinSize((300, 150))
    #sr.SetMaxSize((300, 300))
    fi.AddNewElement(sr)
    p.AddPane(fi)

    #for i in range(4):
    fi = foldingPane(p, -1, caption='pane 4')
    pi = collapsingPane(fi, -1, caption='foo')
    sr = SizeReportCtrl(pi)
    sr.SetMinSize((300, 150))
    pi.AddNewElement(sr)
    #sr.SetMaxSize((300, 300))
    fi.AddNewElement(pi)
    p.AddPane(fi)

    #for i in range(4):
    fi = foldingPane(p, -1, caption='pane 5')
    sr = SizeReportCtrl(fi)
    sr.SetMinSize((300, 150))
    #sr.SetMaxSize((300, 300))
    fi.AddNewElement(sr)
    p.AddPane(fi)

    app.SetTopWindow(f)
    f.Show()
    app.MainLoop()



        
