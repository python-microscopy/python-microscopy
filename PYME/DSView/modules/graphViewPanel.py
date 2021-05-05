#!/usr/bin/python
##################
# graphViewPanel.py
#
# Copyright David Baddeley, 2011
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
import wx.lib.agw.aui as aui
import numpy as np

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import _load_bitmap, error_msg_wx, cursord
#from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.figure import Figure

from PYME.DSView.displayOptions import DisplayOpts

import logging
logger = logging.getLogger(__file__)

        
class MyNavigationToolbar(NavigationToolbar2, aui.AuiToolBar):
    def __init__(self, canvas, wind):
        self.wind = wind
        #wx.ToolBar.__init__(self, canvas.GetParent(), -1)
        aui.AuiToolBar.__init__(self, wind, -1, wx.DefaultPosition, wx.DefaultSize, agwStyle=aui.AUI_TB_DEFAULT_STYLE | aui.AUI_TB_OVERFLOW | aui.AUI_TB_VERTICAL)
        NavigationToolbar2.__init__(self, canvas)
        self.canvas = canvas
        self._idle = True
        self.statbar = None
        

    def get_canvas(self, frame, fig):
        return FigureCanvas(frame, -1, fig)

    def _init_toolbar(self):
        self._parent = self.canvas.GetParent()
        _NTB2_HOME    =wx.NewId()
        self._NTB2_BACK    =wx.NewId()
        self._NTB2_FORWARD =wx.NewId()
        self._NTB2_PAN     =wx.NewId()
        self._NTB2_ZOOM    =wx.NewId()
        _NTB2_SAVE    = wx.NewId()
        _NTB2_SUBPLOT    =wx.NewId()

        self.SetToolBitmapSize(wx.Size(16,16))

        self.AddSimpleTool(_NTB2_HOME,
                           'Home', _load_bitmap('home.png'), 'Reset original view')
        self.AddSimpleTool(self._NTB2_BACK, 
                           'Back', _load_bitmap('back.png'), 'Back navigation view')
        self.AddSimpleTool(self._NTB2_FORWARD, 'Forward', _load_bitmap('forward.png'),
                           'Forward navigation view')
        # todo: get new bitmap
        self.AddCheckTool(self._NTB2_PAN, 'Pan', _load_bitmap('move.png'), _load_bitmap('move.png'),
                           long_help_string='Pan with left, zoom with right')
        self.AddCheckTool(self._NTB2_ZOOM, 'Zoom', _load_bitmap('zoom_to_rect.png'), _load_bitmap('zoom_to_rect.png'),
                           short_help_string='Zoom', long_help_string='Zoom to rectangle')

        self.AddSeparator()
        #self.AddSimpleTool(_NTB2_SUBPLOT, 'Configure subplots', _load_bitmap('subplots.png'),
        #                   'Configure subplot parameters')

        self.AddSimpleTool(_NTB2_SAVE, 'Save', _load_bitmap('filesave.png'),
                           'Save plot contents to file')

        self.wind.Bind(wx.EVT_TOOL, self.home, id=_NTB2_HOME)
        self.wind.Bind(wx.EVT_TOOL, self.forward, id=self._NTB2_FORWARD)
        self.wind.Bind(wx.EVT_TOOL, self.back, id=self._NTB2_BACK)
        self.wind.Bind(wx.EVT_TOOL, self.zoom, id=self._NTB2_ZOOM)
        self.wind.Bind(wx.EVT_TOOL, self.pan, id=self._NTB2_PAN)
        #self.wind.Bind(wx.EVT_TOOL, self.configure_subplot, id=_NTB2_SUBPLOT)
        self.wind.Bind(wx.EVT_TOOL, self.save, id=_NTB2_SAVE)

        self.Realize()


    def zoom(self, *args):
        self.ToggleTool(self._NTB2_PAN, False)
        NavigationToolbar2.zoom(self, *args)

    def pan(self, *args):
        self.ToggleTool(self._NTB2_ZOOM, False)
        NavigationToolbar2.pan(self, *args)


#    def configure_subplot(self, evt):
#        frame = wx.Frame(None, -1, "Configure subplots")
#
#        toolfig = Figure((6,3))
#        canvas = self.get_canvas(frame, toolfig)
#
#        # Create a figure manager to manage things
#        figmgr = FigureManager(canvas, 1, frame)
#
#        # Now put all into a sizer
#        sizer = wx.BoxSizer(wx.VERTICAL)
#        # This way of adding to sizer allows resizing
#        sizer.Add(canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
#        frame.SetSizer(sizer)
#        frame.Fit()
#        tool = SubplotTool(self.canvas.figure, toolfig)
#        frame.Show()

    def save(self, evt):
        import os
        # Fetch the required filename and file type.
        filetypes, exts, filter_index = self.canvas._get_imagesave_wildcards()
        default_file = "image." + self.canvas.get_default_filetype()
        dlg = wx.FileDialog(self._parent, "Save to file", "", default_file,
                            filetypes,
                            wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
        dlg.SetFilterIndex(filter_index)
        if dlg.ShowModal() == wx.ID_OK:
            dirname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            #DEBUG_MSG('Save file dir:%s name:%s' % (dirname, filename), 3, self)
            format = exts[dlg.GetFilterIndex()]
            basename, ext = os.path.splitext(filename)
            if ext.startswith('.'):
                ext = ext[1:]
            if ext in ('svg', 'pdf', 'ps', 'eps', 'png') and format!=ext:
                #looks like they forgot to set the image type drop
                #down, going with the extension.
                #warnings.warn('extension %s did not match the selected image type %s; going with %s'%(ext, format, ext), stacklevel=0)
                format = ext
            try:
                self.canvas.print_figure(
                    os.path.join(dirname, filename), format=format)
            except Exception as e:
                #print str(e)
                error_msg_wx(str(e))

    def set_cursor(self, cursor):
        cursor =wx.StockCursor(cursord[cursor])
        self.canvas.SetCursor( cursor )

    def release(self, event):
        try: del self.lastrect
        except AttributeError: pass

    def dynamic_update(self):
        d = self._idle
        self._idle = False
        if d:
            self.canvas.draw()
            self._idle = True

    def draw_rubberband(self, event, x0, y0, x1, y1):
        'adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/189744'
        canvas = self.canvas
        dc =wx.ClientDC(canvas)

        # Set logical function to XOR for rubberbanding
        dc.SetLogicalFunction(wx.XOR)

        # Set dc brush and pen
        # Here I set brush and pen to white and grey respectively
        # You can set it to your own choices

        # The brush setting is not really needed since we
        # dont do any filling of the dc. It is set just for
        # the sake of completion.

        wbrush =wx.Brush(wx.Colour(255,255,255), wx.TRANSPARENT)
        wpen =wx.Pen(wx.Colour(200, 200, 200), 1, wx.SOLID)
        dc.SetBrush(wbrush)
        dc.SetPen(wpen)


        dc.ResetBoundingBox()
        #dc.BeginDrawing()
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0

        if y1<y0: y0, y1 = y1, y0
        if x1<y0: x0, x1 = x1, x0

        w = x1 - x0
        h = y1 - y0

        rect = int(x0), int(y0), int(w), int(h)
        try: lastrect = self.lastrect
        except AttributeError: pass
        else: dc.DrawRectangle(*lastrect)  #erase last
        self.lastrect = rect
        dc.DrawRectangle(*rect)
        #dc.EndDrawing()

    def set_status_bar(self, statbar):
        self.statbar = statbar

    def set_message(self, s):
        if self.statbar is not None: self.statbar.set_function(s)

    def set_history_buttons(self):
        try:
            can_backward = (self._views._pos > 0)
            can_forward = (self._views._pos < len(self._views._elements) - 1)
            self.EnableTool(self._NTB2_BACK, can_backward)
            self.EnableTool(self._NTB2_FORWARD, can_forward)
        except:
            logger.exception('Error setting history buttons')
        


class GraphViewPanel(wx.Panel):
    def __init__(self, parent, dstack = None, do = None, xvals=None, xlabel='',ylabel=''):
        wx.Panel.__init__(self, parent)

        if (dstack is None and do is None):
            dstack = np.zeros((10,10))

        if do is None:
            self.do = DisplayOpts(dstack)
            self.do.Optimise()
        else:
            self.do = do

        self.do.WantChangeNotification.append(self.draw)

        self.xvals = xvals
        self.xlabel = xlabel
        self.ylabel = ylabel
            
        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)

        sizer1.Add(self.canvas, 1, wx.TOP | wx.LEFT | wx.EXPAND)

        #self.toolbar = NavigationToolbar2WxAgg(self.canvas)
        #self.toolbar = MyNavigationToolbar(self.canvas, self)
        #self.toolbar.Realize()

#        if wx.Platform == '__WXMAC__':
#            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
#            # having a toolbar in a sizer. This work-around gets the buttons
#            # back, but at the expense of having the toolbar at the top
#            self.SetToolBar(self.toolbar)
#        else:
#            # On Windows platform, default window size is incorrect, so set
#            # toolbar width to figure width.
#            tw, th = self.toolbar.GetSizeTuple()
#            fw, fh = self.canvas.GetSizeTuple()
#            # By adding toolbar in sizer, we are able to put it at the bottom
#            # of the frame - so appearance is closer to GTK version.
#            # As noted above, doesn't work for Mac.
#            self.toolbar.SetSize(wx.Size(fw, th))
#            
#            sizer1.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)

        self.Bind(wx.EVT_SIZE, self._onSize)

        #self.toolbar.update()
        self.SetSizer(sizer1)
        self.draw()

    def draw(self, event=None):
        self.axes.cla()

        if self.xvals is None:
            xvals = np.arange(self.do.ds.shape[0])
        else:
            xvals = self.xvals

        n_chans = self.do.ds.shape[self.do.ds.ndim -1]
        
        for i in range(n_chans):
            if self.do.ds.ndim >= 5:
                self.axes.plot(xvals, self.do.ds[:, self.do.yp, self.do.zp, self.do.tp, i].squeeze(), label=self.do.names[i])
            else:
                self.axes.plot(xvals, self.do.ds[:,self.do.yp, self.do.zp, i].squeeze(), label=self.do.names[i])

        if n_chans > 1:
            self.axes.legend()

        self.axes.grid()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        
        self.canvas.draw()

    def _onSize( self, event ):
        #self._resizeflag = True
        self._SetSize()


    def _SetSize( self ):
        pixels = tuple( self.GetClientSize() )
        self.SetSize( pixels )
        self.canvas.SetSize( pixels )
        self.figure.set_size_inches( float( pixels[0] )/self.figure.get_dpi(),
                                     float( pixels[1] )/self.figure.get_dpi() )

def Plug(dsviewer):
    #if dsviewer.image.mdh and 'xvalues' in dsviewer.image.mdh.getEntryNames():
    #    xvals = dsviewer.image.mdh.getEntry('xvalues')
    #    xlabel = dsviewer.image.mdh.getEntry('xlabel')

    if 'xvals' in dir(dsviewer.image):
        xvals = dsviewer.image.xvals
        xlabel = dsviewer.image.xlabel

    else:
        xvals = None
        xlabel = ''

    if 'ylabel' in dir(dsviewer.image):
        ylabel = dsviewer.image.ylabel
    else:
        ylabel=''

        
    dsviewer.gv = GraphViewPanel(dsviewer, do=dsviewer.do, xvals=xvals, xlabel=xlabel,ylabel=ylabel)
    dsviewer.AddPage(dsviewer.gv, True, 'Graph View')
    
    dsviewer.gv.toolbar = MyNavigationToolbar(dsviewer.gv.canvas, dsviewer)
    dsviewer._mgr.AddPane(dsviewer.gv.toolbar, aui.AuiPaneInfo().Name("MPLTools").Caption("Matplotlib Tools").CloseButton(False).
                      ToolbarPane().Right().GripperTop())
