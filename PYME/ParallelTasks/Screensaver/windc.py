"""Simple wrapper around a windows DC (drawing surface)

Note:
dc.beginDrawing()
    This one must be called before any other drawing function!
    Its enough to call it once for a largr block of drawing
    operations. When the drawing is finished endDrawing has to
    be called.
    
dc.endDrawing()
    Finish drawing.

TODO:
- The methods are not implemented verry efficiently but it should be fast
  enough for most applications.
- Many operations that the gdi32 library supplies are not wrapped.
  The user can use ctypes to call the other ones ;-)

(C) 2003 Chris Liechti <cliechti@gmx.net>
This is distributed under a free software license, see license.txt.
"""

from ctypes import *
from winuser import*
from wingdi import *
from wincommon import *

class DC:
    """Drawing are with methods to draw on it."""
    
    def __init__(self, hWnd):
        self.dc = None
        self.hWnd = hWnd
    
    #------------------------------------------------------------------------
    
    def beginDrawing(self):
        """must be called before any drawX or fillX function.
        endDrawing() is called after the paint operations are complete.
        many paint operatios may be used between the beginand endDrawin calls."""
        self.dc = windll.user32.GetDC(self.hWnd)
        
    def endDrawing(self):
        """end drawing, free internal DC resources"""
        windll.user32.ReleaseDC(self.hWnd, self.dc)
    
    #------------------------------------------------------------------------
    # drawing functions
    #------------------------------------------------------------------------
    
    def drawLine(self, (x1,y1), (x2,y2)):
        windll.gdi32.MoveToEx(self.dc, x1, y1, NULL)
        windll.gdi32.LineTo(self.dc, x2, y2)

    def drawText(self, (x,y), text):
        windll.gdi32.TextOutA(self.dc, int(x), int(y), text, len(text))

    def drawRect(self, (x1,y1), (x2,y2)):
        windll.gdi32.MoveToEx(self.dc, x1, y1, NULL);
        windll.gdi32.LineTo(self.dc, x1, y2);
        windll.gdi32.LineTo(self.dc, x2, y2);
        windll.gdi32.LineTo(self.dc, x2, y1);
        windll.gdi32.LineTo(self.dc, x1, y1);
    
    def fillRect(self, (x1,y1), (x2,y2)):
        windll.gdi32.Rectangle(self.dc, x1, y1, x2, y2)
        
    def invertRect(self, (x1,y1), (x2,y2)):
        rc = RECT()
        rc.left = x1
        rc.top = y1
        rc.right = x2
        rc.bottom = y2
        windll.gdi32.InvertRect(self.dc, byref(rc))
    
    def fillEllipse(self, (x1,y1), (x2,y2)):
        windll.gdi32.Ellipse(self.dc, x1, y1, x2, y2)
    
    
    def setColor(self, color):
        hOld = windll.gdi32.SelectObject(self.dc, windll.gdi32.CreatePen(PS_SOLID, 1, color))
        windll.gdi32.DeleteObject(hOld)
    
    def setTextColor(self, color):
        windll.gdi32.SetTextColor(self.dc, color)
    
    def setBgColor(self, color):
        windll.gdi32.SetBkColor(self.dc, color)
    
    def setBgTransparent(self, transparent=False):
        windll.gdi32.SetBkMode(self.dc, transparent and TRANSPARENT or OPAQUE)
    
    def setFillColor(self, color):
        hOld = windll.gdi32.SelectObject(self.dc, windll.gdi32.CreateSolidBrush(color))
        windll.gdi32.DeleteObject(hOld)
    
    def setFont(self, fontname, size=20, bold=False, italic=False, rotation=0):
        lf = LOGFONT()

        lf.lfHeight = int(size)
        lf.lfWidth = 0
        lf.lfEscapement = int(rotation)
        lf.lfOrientation = int(rotation)
        lf.lfWeight = bold and FW_BOLD or FW_NORMAL
        lf.lfItalic = italic
        lf.lfUnderline = False
        lf.lfStrikeOut = False
        lf.lfCharSet = ANSI_CHARSET
        lf.lfOutPrecision = OUT_DEFAULT_PRECIS
        lf.lfClipPrecision = CLIP_DEFAULT_PRECIS
        lf.lfQuality = ANTIALIASED_QUALITY
        lf.lfPitchAndFamily = DEFAULT_PITCH
        lf.lfFaceName = fontname

        hOld = windll.gdi32.SelectObject(self.dc, windll.gdi32.CreateFontIndirectA(byref(lf)))
        windll.gdi32.DeleteObject(hOld)

    #------------------------------------------------------------------------

    def getSize(self):
        """return a tuple with the with and height of the drawing area"""
        rc = RECT()
        windll.user32.GetClientRect(self.hWnd, byref(rc));
        return (rc.right, rc.bottom)
