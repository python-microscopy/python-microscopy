#!/usr/bin/python

###############
# positionTracker.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################
import time
import numpy as np
import wx

from PYME.contrib.wxPlotPanel import PlotPanel

class TrackerPlotPanel(PlotPanel):
    def __init__(self, parent, posTrk, *args, **kwargs):
        self.posTrk = posTrk
        PlotPanel.__init__(self, parent, *args, **kwargs)

        

    def draw(self):
        if self.IsShownOnScreen():
            if not hasattr( self, 'subplot' ):
                    self.subplot = self.figure.add_subplot( 111 )
    
            try:
                trk  = self.posTrk.GetTrack()
    
                self.subplot.cla()
                
                #circle2=plt.Circle((5,5),.5,color='b',fill=False)
    
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
    def __init__(self, scope, time1, viewsize=25., nPixels=500):
        self.track = []
        #self.stds = []
        self.tags = {}
        self.tags['edge'] = []

        self.scope = scope
        self.time1 = time1
        self.im = np.zeros([nPixels, nPixels])
        self.ps = viewsize/nPixels

        time1.register_callback(self.Tick)

    def Tick(self):
        positions  = np.zeros(len(self.scope.piezos) + 2)
        positions[0] = time.time()
        
        for i, p in enumerate(self.scope.piezos):
            if 'GetLastPos' in dir(p[0]):
                positions[i+1] = p[0].GetLastPos(p[1])
            elif 'lastPos' in dir(p[0]):
                positions[i+1] = p[0].lastPos
            else:
                positions[i+1] = p[0].GetPos(p[1])
                
        positions[-1] = np.std(self.scope.frameWrangler.currentFrame)
        
        t, z, x, y, s = positions
        xi = int(np.round(x/self.ps))
        yi = int(np.round(y/self.ps))
        self.im[xi, yi] = max(self.im[xi, yi], s)

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
        
    def GetImage(self, viewsize=25., nPixels=500):
        im = np.zeros([nPixels, nPixels])
        
        ps = viewsize/nPixels
        for pos in self.track:
            t, x, y, z, s = pos
            xi = int(np.round(x/ps))
            yi = int(np.round(y/ps))
            im[xi, yi] = max(im[xi, yi], s)
            
        return im



