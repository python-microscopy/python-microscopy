import wx
import seqpanel

class sqPanel(seqpanel.sPanel):
    def __init__(self, *args, **kwds):
        seqpanel.sPanel.__init__(self, *args, **kwds)

        wx.EVT_BUTTON(self, self.bStart.GetId(), self.onStart)
        

    def onStart(self, event):
        self.scope.pa.stop()

        self.scope.sa = simplesequenceaquisator.SimpleSequenceAquisitor(self.chaninfo, self.cam, piezo)
        self.scope.sa.SetStartMode(self.scope.sa.START_AND_END)

        self.scope.sa.SetStepSize(stepsize)
        self.scope.sa.SetStartPos(endpos)
        self.scope.sa.SetEndPos(startpos)

        self.scope.sa.Prepare()

        
        self.scope.sa.WantFrameNotification.append(self.scope.aq_refr)

        self.scope.sa.WantStopNotification.append(self.scope.aq_end)

        self.scope.sa.start()

        self.scope.pb = wx.ProgressDialog('Aquisition in progress ...', 'Slice 1 of %d' % self.sa.ds.getDepth(), self.sa.ds.getDepth(), style = wx.PD_APP_MODAL|wx.PD_AUTO_HIDE|wx.PD_REMAINING_TIME|wx.PD_CAN_ABORT)