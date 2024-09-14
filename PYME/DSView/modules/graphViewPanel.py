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
from matplotlib.backends.backend_wx import _load_bitmap
#from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.figure import Figure

from PYME.DSView.displayOptions import DisplayOpts

import logging
logger = logging.getLogger(__file__)

        
class MyNavigationToolbar(NavigationToolbar2, aui.AuiToolBar):
    """ This is a customised version of the matplotlib NavigationToolbar2WxAgg, with
    the main difference being that it is an AUI toolbar, rather than a vanilla wx toolbar.
    
    This should ideally track changes in the matplotlib version of NavigationToolbar2WxAgg,
    so if anything breaks with matplotlib updates, that's the first place to look.
    """
    def __init__(self, canvas, wind):
        self.wind = wind
        #wx.ToolBar.__init__(self, canvas.GetParent(), -1)
        aui.AuiToolBar.__init__(self, wind, -1, wx.DefaultPosition, wx.DefaultSize, agwStyle=aui.AUI_TB_DEFAULT_STYLE | aui.AUI_TB_OVERFLOW | aui.AUI_TB_VERTICAL)
        
        self.canvas = canvas
        self._idle = True
        self.statbar = None
        self._init_toolbar()
        
        NavigationToolbar2.__init__(self, canvas)
           

    def get_canvas(self, frame, fig):
        return FigureCanvas(frame, -1, fig)

    def _init_toolbar(self):
        self._parent = self.canvas.GetParent()
        _NTB2_HOME    =wx.NewIdRef()
        self._NTB2_BACK    =wx.NewIdRef()
        self._NTB2_FORWARD =wx.NewIdRef()
        self._NTB2_PAN     =wx.NewIdRef()
        self._NTB2_ZOOM    =wx.NewIdRef()
        _NTB2_SAVE    = wx.NewIdRef()
        _NTB2_SUBPLOT    =wx.NewIdRef()

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

    def save(self, evt):
        # TODO - hijack save button to let us save graph data as well as the dispaled version???
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
                logger.exception('Error saving figure')
                # TODO - graphical error?
                #error_msg_wx(str(e))

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        self.canvas._rubberband_rect = (x0, height - y0, x1, height - y1)
        self.canvas.Refresh()

    def remove_rubberband(self):
        self.canvas._rubberband_rect = None
        self.canvas.Refresh()
        


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
