#!/usr/bin/python

##################
# viewer.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from wxPython.wx import *
import os

try:
    import Numeric
except:
    try:
        import numarray as Numeric  
    except:
        msg= """
        This module requires the Numeric or numarray module,
        which could not be imported.  It probably is not installed
        (it's not part of the standard Python distribution). See the
        Python site (http://www.python.org) for information on
        downloading source or binaries."""
        raise ImportError, "Numeric or numarray not found. \n" + msg

#--------------------------------------------------------------------------------

class PolyPoints:

    def __init__(self, points, attr):
        self.points = Numeric.array(points)
        self.scaled = self.points
        self.attributes = {}
        for name, value in self._attributes.items():
            try:
                value = attr[name]
            except KeyError: pass
            self.attributes[name] = value

    def boundingBox(self):
        minXY=Numeric.minimum.reduce(self.points)
        maxXY=Numeric.maximum.reduce(self.points)
        return minXY, maxXY

    def scaleAndShift(self, scale=1, shift=0):
        self.scaled = scale*self.points+shift

#--------------------------------------------------------------------------------

class PolyLine(PolyPoints):

    def __init__(self, points, **attr):
        PolyPoints.__init__(self, points, attr)

    _attributes = {'color': 'red',
                   'width': 1}

    def draw(self, dc):
        color = self.attributes['color']
        width = self.attributes['width']
        arguments = []
        dc.SetPen(wx.wxPen(wx.wxNamedColour(color), width))
        dc.DrawLines(map(tuple,self.scaled))

#--------------------------------------------------------------------------------

class PlotGraphics:

    def __init__(self, objects):
        self.objects = objects

    def boundingBox(self):
        p1, p2 = self.objects[0].boundingBox()
        for o in self.objects[1:]:
            p1o, p2o = o.boundingBox()
            p1 = Numeric.minimum(p1, p1o)
            p2 = Numeric.maximum(p2, p2o)
        return p1, p2

    def scaleAndShift(self, scale=1, shift=0):
        for o in self.objects:
            o.scaleAndShift(scale, shift)

    def draw(self, canvas):
        for o in self.objects:
            o.draw(canvas)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, item):
        return self.objects[item]

#--------------------------------------------------------------------------------

class PlotCanvas(wx.wxWindow):

    def __init__(self, parent, id=-1,
                 pos = wxPoint(0, 29), size = wx.wxDefaultSize,
                 style = wxSUNKEN_BORDER, name = 'plotCanvas'):
                    
        wx.wxWindow.__init__(self, parent, id, pos, size, style, name)
        self.SetClientSizeWH(600,571)
        
        EVT_PAINT(self, self.OnPaint)
        
        OriginX=parent.OriginX
        OriginY=parent.OriginY
        boxsize=parent.boxsize

        self.scale=(1,1)
        self.shift=(0,0)
        
        self.LU=[0,0]
        self._setsize(OriginX, OriginY, boxsize)
        self.last_draw = None

#--------------------------------------------------------------------------------
        
    def OnPaint(self, event):
        pdc = wx.wxPaintDC(self)
        if self.last_draw is not None:
            apply(self.draw, self.last_draw + (pdc,))
    
    def _testFont(self, font):
        if font is not None:
            bg = self.canvas.cget('background')
            try:
                item = CanvasText(self.canvas, 0, 0, anchor=NW,
                                  text='0', fill=bg, font=font)
                self.canvas.delete(item)
            except TclError:
                font = None
        return font

    def _setsize(self, OriginX, OriginY, size):
        self.plotbox_size = Numeric.array([size, -1*size])
        self.plotbox_origin = Numeric.array([OriginX, size+OriginY])

#--------------------------------------------------------------------------------

    def draw(self, graphics, xaxis = None, yaxis = None, dc = None):
        if dc == None: dc = wx.wxClientDC(self)
        dc.BeginDrawing()
        dc.Clear()
        self.last_draw = (graphics, xaxis, yaxis)

        p1, p2 = graphics.boundingBox()
        xticks = self._ticks(p1[0],p2[0])
        yticks = self._ticks(p1[1],p2[1])
        xaxis = self._axisInterval(xaxis, p1[0], p2[0])
        yaxis = self._axisInterval(yaxis, p1[1], p2[1])
        if xaxis is not None:
            p1[0] = xaxis[0]
            p2[0] = xaxis[1]
            xticks = self._ticks(xaxis[0], xaxis[1])
        else:
            xticks = None
        if yaxis is not None:
            p1[1] = yaxis[0]
            p2[1] = yaxis[1]
            yticks = self._ticks(yaxis[0], yaxis[1])
        else:
            yticks = None

        scale = (self.plotbox_size-Numeric.array([0.,0.])) / (p2-p1)
        shift = -p1*scale + self.plotbox_origin
        self.scale=scale
        self.shift=shift

        ox=self.plotbox_origin[0]
        oy=self.plotbox_origin[1]+self.plotbox_size[1]
        w=self.plotbox_size[0]
        h=-1*self.plotbox_size[1]
        dc.SetPen(wx.wxPen(wx.wxNamedColour('BLACK'),2,wxSOLID))
        dc.DrawLine(ox,oy,ox,oy+h)
        dc.DrawLine(ox,oy,ox+w,oy)
        dc.DrawLine(ox+w,oy+h,ox+w,oy)
        dc.DrawLine(ox+w,oy+h,ox,oy+h)

        self._drawAxes(dc, p1, p2, xticks, yticks, ox, oy, w, h)

        self.LU[0]=min(p1[0],xticks[0][0])
        self.LU[1]=max(p2[0],xticks[-1][0])

        graphics.scaleAndShift(scale, shift)
        graphics.draw(dc)
        dc.EndDrawing()
        
    def ShowCoord(self, X, Y, X1, Y1, color='black', dc=None):
        if dc == None:dc = wx.wxClientDC(self)
        self.redraw()
        dc.BeginDrawing()
        all=[]
        all.append((X,Y,'black'))
        all.append((X1,Y1,'yellow'))
        for o in all:
            for i in range(len(o[0])):
                p1=(o[0][i],o[1][i])
                p2=self.scale*p1+self.shift
                dc.SetPen(wx.wxPen(wx.wxNamedColour(o[2]), width=1))
                a=int(p2[0])
                b=int(p2[1])
                dc.DrawLine(a-5,b-5,a+5,b+5)
                dc.DrawLine(a-5,b+5,a+5,b-5)
                coord='['+str(o[0][i])+','+str(o[1][i])+']'
                w, h=dc.GetTextExtent(coord)
                dc.SetTextForeground(wx.wxNamedColour(o[2]))
                dc.DrawText(coord,p2[0]-0.5*w,p2[1]-h-10)

        dc.EndDrawing
        
    def redraw(self,dc=None):
        if self.last_draw is not None:
            apply(self.draw, self.last_draw + (dc,))
    
#--------------------------------------------------------------------------------

    def _axisInterval(self, spec, lower, upper):
        if spec is None:
            return None

        if spec == 'minimal':
            if lower == upper:
                return lower-0.5, upper+0.5
            else:
                return lower, upper

        if spec == 'automatic':
            range = upper-lower
            if range == 0.:
                return lower-0.5, upper+0.5
            log = Numeric.log10(range)
            power = Numeric.floor(log)
            fraction = log-power
            if fraction <= 0.05:
                power = power-1
            grid = 10.**power
            lower = lower - lower % grid
            mod = upper % grid
            if mod != 0:
                upper = upper - mod + grid
            return lower, upper

        if type(spec) == type(()):
            lower, upper = spec
            if lower <= upper:
                return lower, upper
            else:
                return upper, lower
        raise ValueError, str(spec) + ': illegal axis specification'

    def _drawAxes(self, dc, p1, p2, xticks, yticks, ox, oy, w, h):
        dc.SetPen(wx.wxPen(wx.wxNamedColour('BLACK'),1,wxLONG_DASH))
        yoff=0.5*dc.GetTextExtent(yticks[0][1])[1]
        if xticks[-1][0]>p2[0] or xticks[0][0]<p1[0]:
            xdiff=xticks[-1][0]-xticks[0][0]
        else:
            xdiff=p2[0]-p1[0]
            
        if yticks[-1][0]>p2[1] or yticks[0][0]<p1[1]:
            ydiff=yticks[-1][0]-yticks[0][0]
        else:
            ydiff=p2[1]-p1[1]
            
        xa=min(p1[0],xticks[0][0])
        ya=min(p1[1],yticks[0][0])

        for x, label in xticks:
            xp=w*(x-xa)/xdiff+ox
            xoff=0.5*dc.GetTextExtent(label)[0]
            dc.DrawLine(xp,oy,xp,oy+h)
            dc.DrawText(label,xp-xoff,oy+h)
        
        for y, label in yticks:
            yp=h-h*(y-ya)/ydiff+oy
            dc.DrawLine(ox,yp,ox+w,yp)
            xoff=dc.GetTextExtent(label)[0]
            dc.DrawText(label,ox-xoff,yp-yoff)
        
        if xticks[0][0]>p1[0]:
            xoff=0.5*dc.GetTextExtent(str(p1[0]))[0]
            dc.DrawText(str(p1[0]),ox-xoff,oy+h)
        if xticks[-1][0]<p2[0]:
            xoff=0.5*dc.GetTextExtent(str(p2[0]))[0]
            dc.DrawText(str(p2[0]),ox+w-xoff,oy+h)
        
        if yticks[0][0]>p1[1]:
            xoff=dc.GetTextExtent(str(p1[1]))[0]
            dc.DrawText(str(p1[1]),ox-xoff,oy+h-yoff)
        if yticks[-1][0]<p1[1]:
            xoff=dc.GetTextExtent(str(p2[1]))[0]
            dc.DrawText(str(p2[1]),ox-xoff,oy-yoff)
        
    def _ticks(self, lower, upper):
        ideal = (upper-lower)/7.
        log = Numeric.log10(ideal)
        power = Numeric.floor(log)
        fraction = log-power
        factor = 1.
        error = fraction
        for f, lf in self._multiples:
            e = Numeric.fabs(fraction-lf)
            if e < error:
                error = e
                factor = f
        grid = factor * 10.**power
        if power > 3 or power < -3:
            format = '%+7.0e'
        elif power >= 0:
            digits = max(1, int(power))
            format = '%' + `digits`+'.0f'
        else:
            digits = -int(power)
            format = '%'+`digits+2`+'.'+`digits`+'f'
        ticks = []
        t = -grid*Numeric.floor(-lower/grid)
        while t <= upper:
            ticks.append( (t, format % (t,)) )
            t = t + grid
        return ticks

    _multiples = [(2., Numeric.log10(2.)), (5., Numeric.log10(5.))]
    
#--------------------------------------------------------------------------------