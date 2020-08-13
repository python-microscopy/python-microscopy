"""

Base object for dsviewer plugins.

TODO - Adapt for VisGUI / make dsviewer and VisGUI plugins use the same base class

"""
import weakref

class Plugin(object):
    """
    Base class for plugins. Used when the plugin needs to persist anything (has settings etc ...)
    or needs to be accessed by other plugins.
    
    Also keeps weak proxies for the dsviewer object and the do and image properties.
    """
    
    def __init__(self, dsviewer):
        self._dsviewer = weakref.ref(dsviewer)
    
    @property
    def dsviewer(self):
        return self._dsviewer() # type: PYME.DSView.dsviewer
        
    @property
    def do(self):
        return self.dsviewer.do # type: PYME.DSView.displayOptions.DisplayOpts
    
    @property
    def image(self):
        return self.dsviewer.image # type: PYME.IO.image.ImageStack
    
    @property
    def view(self):
        return self.dsviewer.view # type: PYME.DSView.arrayViewPanel.ArrayViewPanel
        
    