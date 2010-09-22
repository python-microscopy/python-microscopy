import time
import numpy as np
import wx

from PYME.misc.wxPlotPanel import PlotPanel

class TrackerPlotPanel(PlotPanel):
    def __init__(self, parent, posTrk, *args, **kwargs):
        self.posTrk = posTrk
        PlotPanel.__init__(self, parent, *args, **kwargs)

        

    def draw(self):
        if not hasattr( self, 'subplot' ):
                self.subplot = self.figure.add_subplot( 111 )

        try:
            trk  = self.posTrk.GetTrack()

            self.subplot.cla()

            self.subplot.plot(trk[:,2], trk[:,3], 'x-')
            self.subplot.plot(trk[-1,2], trk[-1,3], '.r')
            self.subplot.axis('equal')

            for tn in self.posTrk.tags.keys():
                tg = self.posTrk.GetTags(tn)
                self.subplot.plot(tg[:,2], tg[:,3], 'og', ms=4)

        except:
            pass

        #self.subplot.set_xlim(0, 512)
        #self.subplot.set_ylim(0, 256)

        self.canvas.draw()

class TrackerPanel(wx.Panel):
    def __init__(self, parent, posTrk):
        wx.Panel.__init__(self, parent)
        self.posTrk = posTrk

        hsizer = wx.BoxSizer(wx.VERTICAL)
        self.tpan = TrackerPlotPanel(self, posTrk)
        hsizer.Add(self.tpan, 1, wx.EXPAND, 0)

        vsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.bClearTrack = wx.Button(self, -1, 'Clear')
        self.bClearTrack.Bind(wx.EVT_BUTTON, self.OnClear)

        vsizer.Add(self.bClearTrack, 0, wx.ALL, 5)

        self.bTagEdge = wx.Button(self, -1, 'Tag Edge')
        self.bTagEdge.Bind(wx.EVT_BUTTON, self.OnTagEdge)

        vsizer.Add(self.bTagEdge, 0, wx.ALL, 5)

        hsizer.Add(vsizer, 0, 0, 0)

        self.SetSizerAndFit(hsizer)

    def OnClear(self, event):
        self.posTrk.ClearTrack()

    def OnTagEdge(self, event):
        self.posTrk.Tag('edge')

    def draw(self):
        self.tpan.draw()

class PositionTracker:
    def __init__(self, scope, time1):
        self.track = []
        self.tags = {}
        self.tags['edge'] = []

        self.scope = scope
        self.time1 = time1

        time1.WantNotification.append(self.Tick)

    def Tick(self):
        positions  = np.zeros(len(self.scope.piezos) + 1)
        positions[0] = time.time()
        
        for i, p in enumerate(self.scope.piezos):
            if 'GetLastPos' in dir(p[0]):
                positions[i+1] = p[0].GetLastPos(p[1])
            elif 'lastPos' in dir(p[0]):
                positions[i+1] = p[0].lastPos
            else:
                positions[i+1] = p[0].GetPos(p[1])

#        if len(self.track) > 0:
#            print np.absolute(self.track[-1][1:] - positions[1:]), (np.absolute(self.track[-1][1:] - positions[1:]) > [.1, .001, .001]).any()


        if len(self.track) == 0 or (np.absolute(self.track[-1][1:] - positions[1:]) > 0).any():
            self.track.append(positions)

    def Tag(self, tagName='edge'):
        positions  = np.zeros(len(self.scope.piezos) + 1)
        positions[0] = time.time()

        for i, p in enumerate(self.scope.piezos):
            if 'GetLastPos' in dir(p[0]):
                positions[i+1] = p[0].GetLastPos(p[1])
            elif 'lastPos' in dir(p[0]):
                positions[i+1] = p[0].lastPos
            else:
                positions[i+1] = p[0].GetPos(p[1])

        self.tags[tagName].append(positions)

    def ClearTrack(self):
        for i in range(len(self.track)):
            self.track.pop(0)

        for tn in self.tags.keys():
            self.tags[tn] = []

    def Detatch(self):
        self.time1.WantNotification.remove(self.Tick)

    def GetTrack(self):
        return np.vstack(self.track)

    def GetTags(self, tagName='edge'):
        return np.vstack(self.tags[tagName])



