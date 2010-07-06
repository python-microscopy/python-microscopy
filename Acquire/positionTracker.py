import time
import numpy as np

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
        except:
            pass

        #self.subplot.set_xlim(0, 512)
        #self.subplot.set_ylim(0, 256)

        self.canvas.draw()

class PositionTracker:
    def __init__(self, scope, time1):
        self.track = []
        self.tags = []

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

    def Detatch(self):
        self.time1.WantNotification.remove(self.Tick)

    def GetTrack(self):
        return np.vstack(self.track)


