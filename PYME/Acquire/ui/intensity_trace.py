
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
        
        # over-allocate buffer so that there is room at the end for us to record multiple values before displaying. 
        # Overallocation is designed to give a factor of 2 margin at a maximum frame rate of around 1000 hz assuming 
        # display and buffer shifting happens at 2Hz with the standard GUI timer
        self.intensity_avg = np.zeros(n_frames + 1000, dtype=float)
        FastGraphPanel.__init__(self, parent, winid, self.frame_vals, 
                                self.intensity_avg[:n_frames])
        self.wrangler = frame_wrangler
        
        self._mf = weakref.ref(parent)  # weak ref to MainFrame
        self._relative_val = 1

        self._buf_idx=0
        self._n_frames = n_frames

    
    @property
    def do(self):
        """
        get the display options. This is done lazily
        because it might not be created when this panel is instantiated.
        """
        return self._mf().vp.do  # PYME.UI.displayOps.DisplayOptions

    def clear(self):
        """
        Clear the trace
        """
        self._relative_val = 1.
        x0, x1, y0, y1, _, _ = self.do.sorted_selection
        self.intensity_avg = np.ones_like(self.intensity_avg) * np.nan_to_num(np.mean(self.wrangler.currentFrame[x0:x1, y0:y1]))

    def update_trace(self, sender=None, **kwargs):
        """
        Update trace intensity data with the current frame's average value within the selected region (can be called separately from display). 
        This is likely light-weight enough to be hooked to FrameWrangler.onFrame in most circumstances, although could be hooked to a lower-rate timer (e.g. onFrameGroup
        or time1) if desired.
        """
        if self.do.selection.mode != SELECTION_RECTANGLE:
            return
            
        # try to get data from the signal argument (if we bound to the onFrame signal), otherwise use
        # the current frame (if bound to onFrameGroup, time1)
        data = kwargs.get('frameData',self.wrangler.currentFrame).squeeze() 

        #print(data.shape, self.wrangler.currentFrame.shape)
        
        x0, x1, y0, y1, _, _ = self.do.sorted_selection

        # check, do we swap xy / rc here?
        self.intensity_avg[self._buf_idx] = np.nan_to_num(np.mean(data[x0:x1, y0:y1])) / self._relative_val

        if self._buf_idx < (len(self.intensity_avg)-1):
            # prevent an out-of-bounds error if gui bogs down and we don't re-shift buffer in time.
            self._buf_idx += 1

    def update_display(self, sender=None, **kwargs):
        """Updates the GUI display and does buffer shifting.

        Note that this must be wired-up to the GUI timer, not the frameWrangler's onFrame

        Parameters
        ----------
        sender : optional
            dispatch caller, included only to match the required function signature
        """
        
        

        if self._buf_idx >= (self._n_frames - 1):
            #move data backwards in the buffer so that the last data point is at the right hand side of the displayed interval.
            offset = self._buf_idx - (self._n_frames -2)
            #print(self._n_frames, self._buf_idx, offset)
            self.intensity_avg[:self._n_frames] = self.intensity_avg[offset:(offset+ self._n_frames)]
            self._buf_idx = self._n_frames - 1
        
        print(self.intensity_avg[self._n_frames-1])
        self.SetData(self.frame_vals, self.intensity_avg[:self._n_frames])

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
#     scope.frameWrangler.onFrame.connect(intensity_trace.update_trace) # to update data on every frame
#     #scope.frameWrangler.onFrameGroup.connect(intensity_trace.update_trace) # to update data at ~5Hz, or every frame, whichever is slower

#     MainFrame.time1.register_callback(intensity_trace.update_display) # update the display at 2Hz

#     panel = TraceROISelectPanel(MainFrame, intensity_trace)
#     MainFrame.camPanels.append((panel, 'Trace ROI Select'))
