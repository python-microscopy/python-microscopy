"""
Monkey patch for traitsui.wx.constants, to fix a bug whereby text fields are not usable
in dark mode due to a hard-coded background color.

We do this by installing an import hook that will redirect imports of traitsui.wx.constants to
a local version which uses a more appropriate background color.
"""

import sys
import os.path

from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location

class MyMetaFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if 'traitsui.wx.constants' in fullname:
            #print(fullname, path, target)

            return spec_from_file_location(fullname, os.path.join(os.path.dirname(__file__), 'traitsui_constants.py'))

        return None # we don't know how to import this
    
sys.meta_path.insert(0, MyMetaFinder())

# also patch wx aui to use a more appropriate background colour
from wx.lib.agw.aui import aui_utilities
def GetBaseColour():
    import wx
    
    base_colour = wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DFACE)

    # the base_colour is too pale to use as our base colour,
    # so darken it a bit
    if ((255-base_colour.Red()) +
        (255-base_colour.Green()) +
        (255-base_colour.Blue()) < 60):

        base_colour = aui_utilities.StepColour(base_colour, 92)

    return base_colour

aui_utilities.GetBaseColour.__code__ = GetBaseColour.__code__

from wx.lib.agw.aui import tabart
from wx.lib.agw.aui.aui_utilities import StepColour
import wx

class DarkAwareTabArt(tabart.AuiDefaultTabArt):
        
    def SetDefaultColours(self, base_colour=None):
        """
        Sets the default colours, which are calculated from the given base colour.

        :param `base_colour`: an instance of :class:`wx.Colour`. If defaulted to ``None``, a colour
         is generated accordingly to the platform and theme.
        """

        if base_colour is None:
            base_colour = GetBaseColour()

        self.SetBaseColour(base_colour)
        self._border_colour = StepColour(base_colour, 75)
        self._border_pen = wx.Pen(self._border_colour)

        self._background_top_colour = StepColour(base_colour, 90)
        self._background_bottom_colour = StepColour(base_colour, 170)

        self._tab_top_colour = base_colour
        self._tab_bottom_colour = wx.SystemSettings().GetColour(wx.SYS_COLOUR_WINDOW)# wx.WHITE
        self._tab_gradient_highlight_colour = wx.SystemSettings().GetColour(wx.SYS_COLOUR_WINDOW)# wx.WHITE

        self._tab_inactive_top_colour = base_colour
        self._tab_inactive_bottom_colour = StepColour(self._tab_inactive_top_colour, 160)

        self._tab_text_colour = lambda page: page.text_colour
        self._tab_disabled_text_colour = wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)

tabart.AuiDefaultTabArt = DarkAwareTabArt
        