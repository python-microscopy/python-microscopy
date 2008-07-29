import wx
import PreviewAquisator

class PhasePreviewAquisator(PreviewAquisator):
    def __init__(self, _chans, _cam, _ds = None):
        PreviewAquisator.__init__(self, _chans, _cam, _ds)