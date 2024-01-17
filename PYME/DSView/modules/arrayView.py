#!/usr/bin/python
##################
# arrayView.py
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

from PYME.DSView.arrayViewPanel import *
import wx

def save_as_png(view):
    filename = wx.FileSelector('Save current view as', wildcard="PNG files(*.png)|*.png", flags=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
    
    if filename.strip():
        view.GrabPNG(filename)

def save_as_series_of_png(view, filename=None):
    """Save current view to a series of PNG files with z (or t) index as suffix, suitable for use in making a movie
    via ffmpeg or similar tools

    FIXME: Currently only supports z

    Parameters
    ----------
    view :ArrayViewPanel
    filename: str
        filename of the Imagestack used to generate the View. Provided only as a default filestub for the PNG series
    """
    if filename is not None:
        import os
        series_stub = os.path.split(os.path.splitext(filename)[0])[-1]
    filename = wx.FileSelector('Save series of PNGs with the following base name', wildcard="PNG files(*.png)|*.png", flags=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT,
                               default_filename=series_stub)
    if filename.strip():
        view.ExportStackToPNG(filename)

def Plug(dsviewer):
    dsviewer.view = ArrayViewPanel(dsviewer, do=dsviewer.do, voxelsize=lambda : getattr(dsviewer.image, 'voxelsize_nm'))
    dsviewer.updateHooks.append(dsviewer.view.Redraw)
    dsviewer.AddPage(dsviewer.view, True, 'Data')

    dsviewer.AddMenuItem('View', "Copy display to clipboard", lambda e : dsviewer.view.CopyImage())
    dsviewer.AddMenuItem('View', "Save image as PNG", lambda e: save_as_png(dsviewer.view))
    dsviewer.AddMenuItem('View', "Save movie frames as PNG", lambda e: save_as_series_of_png(dsviewer.view, dsviewer.image.filename))

#    dsviewer.overlaypanel = OverlayPanel(dsviewer, dsviewer.view, dsviewer.image.mdh)
#    dsviewer.overlaypanel.SetSize(dsviewer.overlaypanel.GetBestSize())
#    pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
#    dsviewer._mgr.AddPane(dsviewer.overlaypanel, pinfo2)

#    dsviewer._mgr.AddPane(dsviewer.view.CreateToolBar(dsviewer), aui.AuiPaneInfo().Name("ViewTools").Caption("View Tools").CloseButton(False).
#                      ToolbarPane().Right().GripperTop())

    #dsviewer.panesToMinimise.append(pinfo2)


