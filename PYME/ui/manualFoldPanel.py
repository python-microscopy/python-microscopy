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
import time
import numpy as np

import wx.lib.newevent

PanelFoldCommandEvent, EVT_CMD_PANEL_FOLD = wx.lib.newevent.NewCommandEvent()

def ColourFromStyle(col):
    if isinstance(col, six.string_types):
        col = wx.Colour(col)
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
'HAS_PIN' : True,
}

class CaptionButton(object):
    def __init__(self, active_bitmap, inactive_bitmap=None, show_fcn=None, active_fcn = None, onclick=None):
        self._active_bitmap = active_bitmap
        self._inactive_bitmap = inactive_bitmap
        
        self._show_fcn = show_fcn
        self._active_fcn = active_fcn
        self._onclick = onclick
        
        self._rect = [0,0,0,0]
    
    @property
    def size(self):
        return self._active_bitmap.GetWidth(), self._active_bitmap.GetHeight()
    
    @property
    def show(self):
        if self._show_fcn is None:
            return True
        else:
            return self._show_fcn()
        
    @property
    def active(self):
        if self._active_fcn is None:
            return True
        else:
            return self._active_fcn()
        
        
    def click_test(self, event):
        if (self._onclick is not None) and wx.Rect(*self._rect).Contains(event.GetPosition()):
            self._onclick()
            return True
        else:
            return False

class CaptionBar(wx.Window):
    def __init__(self, parent, id = wx.ID_ANY, pos=(-1,-1), caption="",
                 foldIcons=None, cbstyle=DEFAULT_CAPTION_STYLE, pin_bits=pin_bits):

        wx.Window.__init__(self, parent, id, pos=pos,
                           size=(-1,cbstyle['HEIGHT']), style=wx.NO_BORDER)

        self.style = dict(cbstyle)
        self.parent = parent
        self.caption = caption
        self.foldIcons = foldIcons
        
        self.buttons = []

        self._inactive_pin_bitmap = BitmapFromBits(pin_bits, 16, 16, ColourFromStyle(self.style['INACTIVE_PIN_COLOUR']))
        self._active_pin_bitmap = BitmapFromBits(pin_bits, 16, 16, ColourFromStyle(self.style['ACTIVE_PIN_COLOUR']))

        #self.pinButtonRect = (0,0,0,0)
        
        self.buttons.append(CaptionButton(self._active_pin_bitmap, self._inactive_pin_bitmap,
                                          show_fcn=lambda : self.style['HAS_PIN'] and self.parent.foldable,
                                          active_fcn=lambda : self.parent.pinnedOpen,
                                          onclick=lambda : self.parent.TogglePin()))

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftClick)
        #self.Bind(wx.EVT_ENTER_WINDOW, self.OnMouseEnterLeave)
        #self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseEnterLeave)

#        self.Bind(wx.EVT_CHAR, self.OnChar)


    def SetCaption(self, caption):
        self.caption = caption
    
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
        
        icon_width = self.DrawIcon(gc)

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
        gc.DrawText(self.caption, icon_width + self.style['CAPTION_INDENT'], y0)


        # h = self._active_pin_bitmap.GetHeight()
        # w = self._active_pin_bitmap.GetWidth()
        #
        # y0 = self.style['HEIGHT']/2. - h/2.
        #
        # #print wndRect[2]
        #
        # self.pinButtonRect = (wndRect[2] - h - y0, y0, w,h)
        #
        # if self.style['HAS_PIN'] and self.parent.foldable:
        #     if self.parent.pinnedOpen:
        #         gc.DrawBitmap(self._active_pin_bitmap, *self.pinButtonRect)
        #     else:
        #         gc.DrawBitmap(self._inactive_pin_bitmap, *self.pinButtonRect)
        #
        if self.parent.folded and self.parent.foldable:
             self.DrawEllipses(gc)
        
        self.DrawButtons(gc)

        gc.PopState()
        
    def DrawIcon(self, gc):
        # Return width of icon
        return 0
        
    def DrawButtons(self, gc):
        b0 = self.buttons[0]
        
        w, h = b0.size
        y0 = self.style['HEIGHT'] / 2. - h / 2.
        
        x0 = self.GetRect()[2] - y0
        
        for i, b in enumerate(self.buttons):
            if b.show:
                w, h = b.size
                y0 = self.style['HEIGHT'] / 2. - h / 2.
                x0 -= (w + (i>0)*5)
                b._rect = (x0, y0, w,h)
                
                if b.active:
                    gc.DrawBitmap(b._active_bitmap, *b._rect)
                else:
                    gc.DrawBitmap(b._inactive_bitmap, *b._rect)
            
        self._min_x = x0
        
        

    def DrawEllipses(self, gc):
        gc.SetBrush(wx.Brush(ColourFromStyle(self.style['ELLIPSES_COLOUR'])))
        path = gc.CreatePath()
        path.AddCircle(0, 0, self.style['ELLIPSES_RADIUS'])
        path.AddCircle(3*self.style['ELLIPSES_RADIUS'], 0, self.style['ELLIPSES_RADIUS'])
        path.AddCircle(6*self.style['ELLIPSES_RADIUS'], 0, self.style['ELLIPSES_RADIUS'])

        gc.PushState()
        r0 = self.buttons[-1]._rect
        gc.Translate(r0[0], r0[1] + r0[3])
        bx = path.GetBox()
        gc.Translate(-bx[2], -bx[3])

        gc.FillPath(path)

        gc.PopState()

    
    def OnLeftClick(self, event):
        #if wx.Rect(*self.pinButtonRect).Contains(event.GetPosition()):
        #    self.parent.TogglePin()
            
        for b in self.buttons:
            if b.click_test(event):
                rect = self.GetRect()
                self.RefreshRect(rect)
                return
            
        # we didn't hit any buttins, fold the panel instead
        self.parent.ToggleFold()


    def OnSize(self, event):
        rect = self.GetRect()
        self.RefreshRect(rect)



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
        # NOTE: If you are creating a foldingPane to encapsulate other 
        # foldingPanes, the top-level foldingPane must have the flag
        # pinned=True or all the sub-foldingPanes will be hidden,
        # unhidable, and no error will be thrown. Alternatively you
        # can pass folded=False to the top-level pane.
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

        kwargs['style'] = kwargs.get('style', wx.BORDER_SIMPLE)

        
        wx.Panel.__init__(self, * args, ** kwargs)

        if self.orientation == wx.VERTICAL:
            self.sizerflags = wx.EXPAND | wx.ALL
        else:
            self.sizerflags = wx.EXPAND | wx.RIGHT

        self.elements = []

        #without any elements, folding isn't going to do anything
        self.foldable = False
        
        #keep track of the time we were last unfolded to see which pane to fold first
        self._time_last_unfolded = 0

        self.sizer = wx.BoxSizer(self.orientation)

        if self.caption:
            #self.stCaption = wx.StaticText(self, -1, self.caption)
            self.stCaption = self._create_caption_bar()
            #self.stCaption.SetBackgroundColour(wx.TheColourDatabase.FindColour('SLATE BLUE'))
            self.sizer.Add(self.stCaption, 0, wx.EXPAND, 0)

        self.SetSizer(self.sizer)
        
        self.Bind(wx.EVT_SIZE, self.OnSize)
        
    def _create_caption_bar(self):
        """ This is over-rideable in derived classes so that they can implement their own caption bars"""
        return CaptionBar(self, -1, caption=self.caption)
    
    @property
    def can_fold(self):
        return self.foldable and not (self.pinnedOpen or self.folded)

    def SetCaption(self, caption):
        self.caption = caption
        self.stCaption.SetCaption(caption)
        
        return self

    def AddNewElement(self, window, foldable=True, foldedWindow=None, priority=0):
        self.AddElement(foldElement(window, foldable, foldedWindow), priority=priority)

    def AddElement(self, element, priority=0):
        self.elements.append(element)
        
        self.sizer.Add(element.window, priority, self.sizerflags, self.padding)
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
        
    def Pin(self, pin=True):
        if pin:
            self.PinOpen()
        else:
            self.UnPin()
            
        return self

    def TogglePin(self):
        if self.pinnedOpen:
            self.UnPin()
        else:
            self.PinOpen()
            
    def ToggleFold(self):
        if self.folded:
            self.Unfold()
        else:
            self.Fold()
            
        

    def fold1(self, pan=None):
        try:
            self.GetParent().fold1(self)
        except AttributeError:
            pass

    def Fold(self, fold=True):
        if not fold:
            return self.Unfold()
        #print 'foo'
        if not self.folded and not self.pinnedOpen:
            for element in self.elements:
                #print self.folded, 'foo'
                if element.foldable:
                    element.window.Hide()
                    if element.foldedWindow:
                        element.foldedWindow.Show()

            self.folded = True
            try:
                self.stCaption.Refresh()
            except AttributeError:
                pass
            self.Layout()
            wx.PostEvent(self, PanelFoldCommandEvent(self.GetId()))
            self.fold1()
            return True
        else:
            return False
            #self.Layout()

    def Unfold(self):
        #print 'bar'
        if self.folded:
            self._time_last_unfolded = time.time()
            for element in self.elements:
                #print 'bar'
                if element.foldable:
                    element.window.Show()
                    #element.window.Layout()
                    if element.foldedWindow:
                        element.foldedWindow.Hide()

            self.folded = False
            try:
                self.stCaption.Refresh()
            except AttributeError:
                pass
            self.Layout()
            self.fold1()
            wx.PostEvent(self, PanelFoldCommandEvent(self.GetId()))
            return True
        else:
            return False

            #self.Layout()
    
    def OnSize(self, event):
        self.Layout()


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
            
        kwargs['style'] = wx.BORDER_NONE

        #kwargs['folded'] = False

        foldingPane.__init__(self, *args, **kwargs)

        self.stCaption = wx.Panel(self, -1)

        self.bmR = BitmapFromBits(r_arrow, 16, 16, ColourFromStyle('BLACK'))
        self.bmD = BitmapFromBits(d_arrow, 16, 16, ColourFromStyle('BLACK'))

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #self.bFold = wx.BitmapButton(capPan, -1, bitmap=self.bmR, style = wx.NO_BORDER)
        self.bFold = foldButton(self.stCaption, -1)
        self.bFold.SetFolded(self.folded)

        hsizer.Add(self.bFold, 0, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 2)
        hsizer.Add(wx.StaticText(self.stCaption, -1, caption), 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        self.stCaption.SetSizerAndFit(hsizer)
        self.sizer.Add(self.stCaption, 0, 0, 0)

        self.bFold.Bind(wx.EVT_LEFT_UP, self.OnFold)

    def OnFold(self, event=None):
        print('fold')
        if self.folded:
            self.GetParent()._time_last_unfolded=time.time()
            self.Unfold()
            self.bFold.SetFolded(False)
        else:
            self.Fold()
            self.bFold.SetFolded(True)

        #self.Layout()
        #self.Fit()
        self.fold1()




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

        self._stretch_sizer = kwargs.pop('bottom_spacer', True)
        self._one_pane_active = kwargs.pop('single_active_pane', False)

        wx.Panel.__init__(self, *args, **kwargs)

        if self.orientation == wx.VERTICAL:
            self.sizerflags = wx.EXPAND #| wx.BOTTOM
        else:
            self.sizerflags = wx.EXPAND #| wx.RIGHT

        self.priorities = []
        self.panes = []

        self.sizer = wx.BoxSizer(self.orientation)
        self.SetSizer(self.sizer)

        self._in_fold1 = False
        
        self.fold_signal = dispatch.Signal()
        
        self.Bind(wx.EVT_SIZE, self.OnResize)
        self.Bind(EVT_CMD_PANEL_FOLD, self.OnResize)

        #self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeave)

    def AddPane(self, window, priority=0):
        self.panes.append(window)
        self.priorities.append(priority)

        #window.Bind(wx.EVT_ENTER_WINDOW, self.OnMouseEnterPane)
        #window.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeavePane)

        self.RegenSizer()

    def RegenSizer(self):
        self.sizer.Clear()

        for pane, priority in zip(self.panes, self.priorities):
            self.sizer.Add(pane, priority, self.sizerflags, self.padding)

        self._calc_min_max_sizes()

        if self._stretch_sizer:
            self.sizer.AddStretchSpacer()

        self.sizer.Layout()

    def Clear(self):
        self.priorities = []
        for p in self.panes:
            p.Destroy()
        self.panes = []

        self.RegenSizer()

    def _calc_min_max_sizes(self):
        #remember current states
        _state = [p.folded for p in self.panes]

        #prevent fold1 logic from running
        self._in_fold1 = True

        #fold all panes
        for p in self.panes:
            if p.foldable:
                p.Fold()
        
        self.sizer.Layout()
        self.SetMinSize((-1, self.sizer.GetMinSize()[1]))

        #expand all panes
        #for p in self.panes:
        #    #if p.foldable:
        #    p.Unfold()

        #self.sizer.Layout()
        
        
        #self.SetMaxSize((-1, self.sizer.GetMinSize()[1]))  

        #restore inital state
        for p, s in zip(self.panes, _state):
            if not p.folded == s:
                p.Fold(s)

        self._in_fold1 = False
        self.sizer.Layout()  
        
    def fold1(self, pan=None):
        #print('fold1')
        self._in_fold1 = True
        self.Layout()
        #print(self.GetSize(), self.GetBestSize(), self.GetMinSize(), self.GetMaxSize())
        
        if self._one_pane_active and not (pan is None):
            self._collapse_all_other_frames(pan)
        else:
            if (self.GetBestSize()[1] > self.GetSize()[1]):
                #print('collaping old panes')
                self._collapse_old_frames(pan)
        
        
        self.fold_signal.send(sender=self)
        
        self._in_fold1 = False
        self.Refresh()
        
    def _collapse_old_frames(self, pan=None):
        candidates = [p for p in self.panes if (p.can_fold and (not (p == pan)) and (p._time_last_unfolded< (time.time()-1)))]

        #print(candidates)

        if len(candidates) > 0:
            i = np.argmin([p._time_last_unfolded for p in candidates])
            #print i, candidates[i].caption
            candidates[i].Fold()
        
        self.Layout()
        self.Refresh()
        
    def _collapse_all_other_frames(self, pan=None):
        candidates = [p for p in self.panes if (p.foldable and not p.folded and (not (p == pan)) and (p._time_last_unfolded < (time.time() - .1)))]
        
        for c in candidates:
            c.Fold()
            
        self.Layout()
        self.Refresh()
        
    def OnResize(self, event):
        if (not self._in_fold1) and self.IsShownOnScreen():
            self.fold1()

    

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



        
