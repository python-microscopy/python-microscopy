"""A mixin for wxPython windows that allows for cascading layout of parent windows.
This allows easier hiding and showing of nested panels.

This is a cleaner and more easily readable replacement for the previous `fold1()` method in the fold panels.
"""

import logging
logger = logging.getLogger(__name__)

import wx

class CascadingLayoutMixin(object):
    def cascading_layout(self, depth=0):
        """Lays out the parent window in a cascading fashion.
        """
        #print('    '*depth + 'Cascading layout - %s, size = %s' % (self, tuple(self.GetSize()))) 
        self.GetSizer().Fit(self)
        self.SetMinSize(self.GetSize())
        

        #print('    '*depth + 'Cascading layout - %s, new size = %s' % (self,  tuple(self.GetSize())))

        try:
            self.GetParent().cascading_layout(depth + 1)
        except AttributeError:
            try:
                self.GetParent().Layout()
            except AttributeError:
                pass

        self.Layout()
        #print('    '*depth + 'Cascading layout - %s, final size = %s' % (self,  tuple(self.GetSize())))

class CascadingLayoutPanel(wx.Panel, CascadingLayoutMixin):
    """
    A panel that supports cascading layout.
    """
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)