
import wx
import numpy as np
from PYME.ui.fastGraph import FastGraphPanel
import weakref
from PYME.ui.selection import SELECTION_RECTANGLE
import logging

logger = logging.getLogger(__name__)


class IntensityTracePanel(FastGraphPanel):
    def __init__(self, parent, frame_wrangler, winid=-1, n_frames=1000):
        """
        If the square region select tool is active, will plot the average value within it
        over time
        Parameters
        ----------
        
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
        logger.debug('assigning display options')
        self.do = self._mf().vp.do  # PYME.UI.displayOps.DisplayOptions

    def clear(self):
        self._relative_val = 1.
        x0, x1, y0, y1, _, _ = self.do.sorted_selection
        self.intensity_avg = np.ones_like(self.intensity_avg) * np.nan_to_num(np.mean(self.wrangler.currentFrame[x0:x1, y0:y1]))


    def relative_clear(self):
        x0, x1, y0, y1, _, _ = self.do.sorted_selection
        self._relative_val = np.nan_to_num(np.mean(self.wrangler.currentFrame[x0:x1, y0:y1]))
        print('relative value: %f' % self._relative_val)
        self.intensity_avg = np.ones_like(self.intensity_avg)

    def refr(self, sender=None, **kwargs):
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
        
        """
        wx.Panel.__init__(self, parent, winid)
        self.trace_page = trace_page

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.clear_button = wx.Button(self, -1, 'Clear')
        hsizer.Add(self.clear_button, 0, wx.ALL, 2)
        self.clear_button.Bind(wx.EVT_BUTTON, self.OnClear)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        self.relclear_button = wx.Button(self, -1, 'Relative Clear')
        hsizer.Add(self.relclear_button, 0, wx.ALL, 2)
        self.relclear_button.Bind(wx.EVT_BUTTON, self.OnRelClear)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        self.SetSizerAndFit(vsizer)

    def OnClear(self, wx_event=None):
        self.trace_page.clear()

    def OnRelClear(self, wx_event=None):
        self.trace_page.relative_clear()


# example init script plug:
# @init_gui('intensity trace')
# def intensity_trace(MainFrame, scope):
#     from PYME.Acquire.ui.intensity_trace import IntensityTracePanel, TraceROISelectPanel

#     intensity_trace = IntensityTracePanel(MainFrame, scope.frameWrangler)
#     MainFrame.AddPage(page=intensity_trace, select=False, caption='Trace')
#     MainFrame.time1.WantNotification.append(intensity_trace.refr)

#     panel = TraceROISelectPanel(MainFrame, intensity_trace)
#     MainFrame.camPanels.append((panel, 'Trace ROI Select'))
