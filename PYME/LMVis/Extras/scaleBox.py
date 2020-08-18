#!/usr/bin/python

# scaleBox.py
#
# Copyright Michael Graff
#   graff@hm.edu
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
import numpy
import wx

import wx.lib.agw.cubecolourdialog as ccd

from PYME.LMVis.Extras.dockedPanel import DockedPanel


class ScaleBoxPanel(DockedPanel):
    """
    This panel is the gui for creating and controlling the bounding box of the
     data and setting a grid with a given tick distance
    """
    def __init__(self, parent_panel, **kwargs):
        super(ScaleBoxPanel, self).__init__(parent_panel, **kwargs)
        self._show_box = False
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)

        self.create_gui(vertical_sizer)

        self.SetSizerAndFit(vertical_sizer)
        self.show_box()

    def create_gui(self, vertical_sizer):
        x_button = wx.Button(self, -1, label='x', style=wx.BU_EXACTFIT)
        y_button = wx.Button(self, -1, label='y', style=wx.BU_EXACTFIT)
        z_button = wx.Button(self, -1, label='z', style=wx.BU_EXACTFIT)
        color_button = wx.Button(self, -1, label='Color', style=wx.BU_EXACTFIT)
        show_button = wx.Button(self, -1, label='Show', style=wx.BU_EXACTFIT)

        x_button.Bind(wx.EVT_BUTTON, lambda e: self.flip(x_flip=True))
        y_button.Bind(wx.EVT_BUTTON, lambda e: self.flip(y_flip=True))
        z_button.Bind(wx.EVT_BUTTON, lambda e: self.flip(z_flip=True))
        color_button.Bind(wx.EVT_BUTTON, lambda e: self.pick_color())
        show_button.Bind(wx.EVT_BUTTON, lambda e: self.show_box())

        grid_sizer = wx.GridSizer(rows=3, cols=1, vgap=1, hgap=1)
        horizontal_box = wx.BoxSizer(wx.HORIZONTAL)
        horizontal_box.Add(x_button, 1, flag=wx.EXPAND | wx.ALL)
        horizontal_box.Add(y_button, 1, flag=wx.EXPAND | wx.ALL)
        horizontal_box.Add(z_button, 1, flag=wx.EXPAND | wx.ALL)
        grid_sizer.Add(horizontal_box, 1, flag=wx.EXPAND | wx.ALL)
        grid_sizer.Add(color_button, 1, flag=wx.EXPAND | wx.ALL)

        grid_sizer.Add(show_button, 1, flag=wx.EXPAND | wx.ALL)
        vertical_sizer.Add(grid_sizer, 0, flag=wx.EXPAND | wx.ALL)

    def flip(self, x_flip=False, y_flip=False, z_flip=False):
        self.get_scale_box_layer().flip_starts(x_flip, y_flip, z_flip)
        self.refresh_canvas()

    def show_box(self):
        self._show_box = not self._show_box
        self.get_scale_box_layer().show(self._show_box)
        self.refresh_canvas()

    def get_scale_box_layer(self):
        return self.parent_panel.glCanvas.ScaleBoxOverlayLayer

    def refresh_canvas(self):
        #self.parent_panel.glCanvas.OnDraw()
        self.parent_panel.glCanvas.Refresh()

    def pick_color(self):
        """
        This is mostly from the wxPython Demo!
        """
        dlg = ccd.CubeColourDialog(self)

        # Ensure the full colour dialog is displayed,
        # not the abbreviated version.
        dlg.GetColourData().SetChooseFull(True)

        if dlg.ShowModal() == wx.ID_OK:
            data = dlg.GetRGBAColour()
            self.get_scale_box_layer().set_color(numpy.array(data)/255.0)
        dlg.Destroy()
        self.refresh_canvas()


def Plug(vis_fr):
    """
    There's no chance to determine if shaders are supported or not. Since this layer is only
    added to the shader version, it will fail in the non-shader version. The later doesn't
    have the required methods.
    It will fail, when you try to get a bounding box. If you don't select the menu item, the program will run normally
    Parameters
    ----------
    vis_fr

    Returns
    -------

    """
    DockedPanel.add_menu_item(vis_fr, 'Scale Box', ScaleBoxPanel, 'scale_box')
