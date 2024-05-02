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
from wx.lib.agw.aui.aui_utilities import BlendColour
import wx
def StepColour(c, ialpha):
    """
    Darken/lighten the input colour `c`.

    :param wx.Colour `c`: a colour to darken/lighten;
    :param integer `ialpha`: a transparency value.
    """

    if ialpha == 100:
        return c

    r, g, b, a = c.Red(), c.Green(), c.Blue(), c.Alpha()

    if (r + g + b) < (255*3)/2:
        # dark mode
        ialpha = 200 - ialpha

    # ialpha is 0..200 where 0 is completely black
    # and 200 is completely white and 100 is the same
    # convert that to normal alpha 0.0 - 1.0
    ialpha = min(ialpha, 200)
    ialpha = max(ialpha, 0)
    alpha = (ialpha - 100.0)/100.0

    if ialpha > 100:

        # blend with white
        bg = 255
        alpha = 1.0 - alpha  # 0 = transparent fg 1 = opaque fg

    else:

        # blend with black
        bg = 0
        alpha = 1.0 + alpha  # 0 = transparent fg 1 = opaque fg

    r = BlendColour(r, bg, alpha)
    g = BlendColour(g, bg, alpha)
    b = BlendColour(b, bg, alpha)

    return wx.Colour(int(r), int(g), int(b), int(a))

aui_utilities.StepColour.__code__ = StepColour.__code__

def GetBaseColour():
    # this import may not be needed on all wxpython / python versions (but is for 4.1 / 3.7)
    from wx.lib.agw.aui import aui_utilities
    
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

from wx.lib.agw.aui import dockart
from wx.lib.agw.aui.aui_utilities import LightContrastColour

class DarkAwareDockArt(dockart.AuiDefaultDockArt):
    def Init(self):
        """ Initializes the dock art. """

        self.SetDefaultColours()

        isMac = wx.Platform == "__WXMAC__"

        if isMac:
            self._active_caption_colour = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT)
        else:
            self._active_caption_colour = wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION)

        self._active_caption_gradient_colour = LightContrastColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT))
        self._active_caption_text_colour = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHTTEXT)
        self._inactive_caption_text_colour = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)

dockart.AuiDefaultDockArt = DarkAwareDockArt
        