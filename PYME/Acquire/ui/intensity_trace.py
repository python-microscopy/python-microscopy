
import wx
import numpy as np
from PYME.ui.fastGraph import FastGraphPanel
import weakref


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
        self.do = None  # display_op]
        self._mf = weakref.ref(parent)
        self._relative_val = 1

    def assign_do(self):
        print('assigning display options')
        self.do = self._mf().vp.do
        print(self.do.SELECTION_RECTANGLE)

    def clear(self):
        self._relative_val = 1.
        x0, x1, y0, y1 = [self.do.selection_begin_x, self.do.selection_end_x, self.do.selection_begin_y, self.do.selection_end_y]
        self.intensity_avg = np.ones_like(self.intensity_avg) * np.nan_to_num(np.mean(self.wrangler.currentFrame[x0:x1, y0:y1]))


    def relative_clear(self):
        x0, x1, y0, y1 = [self.do.selection_begin_x, self.do.selection_end_x, self.do.selection_begin_y, self.do.selection_end_y]
        self._relative_val = np.nan_to_num(np.mean(self.wrangler.currentFrame[x0:x1, y0:y1]))
        print('relative value: %f' % self._relative_val)
        self.intensity_avg = np.ones_like(self.intensity_avg)

    def refr(self, sender=None, **kwargs):
        if self.do is None:
            try:
                self.assign_do()
            except:
                return
        if self.do.selectionMode != self.do.SELECTION_RECTANGLE:
            return
        x0, x1, y0, y1 = [self.do.selection_begin_x, self.do.selection_end_x, self.do.selection_begin_y, self.do.selection_end_y]

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
# def focus_lock(MainFrame, scope):
#     import numpy as np
#     from XXX import IntensityTracePanel

#     intensity_trace = IntensityTracePanel(MainFrame, scope.frameWrangler)
#     MainFrame.AddPage(page=intensity_trace, select=False, caption='Trace')
#     MainFrame.time1.WantNotification.append(intensity_trace.refr) 