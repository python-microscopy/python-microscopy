
import wx
import numpy as np
from PYME.ui.fastGraph import FastGraphPanel
import weakref
from PYME.ui.selection import SELECTION_RECTANGLE
import logging

logger = logging.getLogger(__name__)


class IntensityTracePanel(FastGraphPanel):
    def __init__(self, parent, frame_wrangler, winid=-1, n_frames=1000):
        """If the square region select tool is active, will plot the average value within it
        over time. Note that the update rate is set by the GUI timer, not the camera frame rate.

        Parameters
        ----------
        parent : wx.Window
            PYMEAcquire MainFrame / AUIFrame
        frame_wrangler : PYME.Acquire.FrameWrangler
            frame_wrangler object, which is grabbing new frames from the camera
        winid : int, optional
            direct pass-through to FastGraphPanel, by default -1
        n_frames : int, optional
            Number of frame-updates to store and display, by default 1000
        """
        self.frame_vals = np.arange(n_frames)
        self.intensity_avg = np.zeros_like(self.frame_vals, dtype=float)
        FastGraphPanel.__init__(self, parent, winid, self.frame_vals, 
                                self.intensity_avg)
        self.wrangler = frame_wrangler
        # differ display opts assignment because it might not have been created yet
        self.do = None  # PYME.UI.displayOps.DisplayOptions gets assigned later
        self._mf = weakref.ref(parent)  # weak ref to MainFrame
        self._relative_val = 1

    def assign_do(self):
        """
        Assign the display options object from the MainFrame. This is done lazily
        because it might not be created when this panel is instantiated.
        """
        logger.debug('assigning display options')
        self.do = self._mf().vp.do  # PYME.UI.displayOps.DisplayOptions

    def clear(self):
        """
        Clear the trace
        """
        self._relative_val = 1.
        x0, x1, y0, y1, _, _ = self.do.sorted_selection
        self.intensity_avg = np.ones_like(self.intensity_avg) * np.nan_to_num(np.mean(self.wrangler.currentFrame[x0:x1, y0:y1]))

    def refr(self, sender=None, **kwargs):
        """Updates the trace with the current frame's average value within the selected region

        Note that this will be wired-up to the GUI timer, not the frameWrangler's update rate

        Parameters
        ----------
        sender : optional
            dispatch caller, included only to match the required function signature
        """
        if self.do is None:
            try:
                self.assign_do()
            except:
                return
        if self.do.selection.mode != SELECTION_RECTANGLE:
            return
        x0, x1, y0, y1, _, _ = self.do.sorted_selection

        self.intensity_avg[:-1] = self.intensity_avg[1:]
        # check, do we swap xy / rc here?
        self.intensity_avg[-1] = np.nan_to_num(np.mean(self.wrangler.currentFrame[x0:x1, y0:y1])) / self._relative_val
        print(self.intensity_avg[-1])
        self.SetData(self.frame_vals, self.intensity_avg)

class TraceROISelectPanel(wx.Panel):
    def __init__(self, parent, trace_page, winid=-1):
        """
        Simple panel to facilitate clearing the Itensity trace through the GUI
        """
        wx.Panel.__init__(self, parent, winid)
        self.trace_page = trace_page

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.clear_button = wx.Button(self, -1, 'Clear')
        hsizer.Add(self.clear_button, 0, wx.ALL, 2)
        self.clear_button.Bind(wx.EVT_BUTTON, self.OnClear)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        self.SetSizerAndFit(vsizer)

    def OnClear(self, wx_event=None):
        self.trace_page.clear()


# example init script plug:
# @init_gui('intensity trace')
# def intensity_trace(MainFrame, scope):
#     from PYME.Acquire.ui.intensity_trace import IntensityTracePanel, TraceROISelectPanel

#     intensity_trace = IntensityTracePanel(MainFrame, scope.frameWrangler)
#     MainFrame.AddPage(page=intensity_trace, select=False, caption='Trace')
#     MainFrame.time1.WantNotification.append(intensity_trace.refr)

#     panel = TraceROISelectPanel(MainFrame, intensity_trace)
#     MainFrame.camPanels.append((panel, 'Trace ROI Select'))
