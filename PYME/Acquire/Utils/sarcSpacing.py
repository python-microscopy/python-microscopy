#!/usr/bin/python

##################
# sarcSpacing.py
#
# Copyright David Baddeley, 2009
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
##################

# import pylab
import matplotlib.pyplot as plt
#from PYME import cSMI
import numpy as np
import wx


class SarcomereChecker:
    def __init__(self, parent, menu, scope, key = 'F12'):
        self.scope = scope

        idSarcCheck = wx.NewIdRef()

        self.menu = wx.Menu(title = '')

        self.menu.Append(idSarcCheck, 'Check Sarcomere Spacing\t%s' % key)
        wx.EVT_MENU(parent, idSarcCheck, self.OnCheckSpacing)


        menu.Append(menu=self.menu, title = '&Utils')
        self.mbar = menu
        self.mpos = menu.GetMenuCount() - 1


    def OnCheckSpacing(self,event):
        voxx = 0.07
        voxy = 0.07
        
        im = self.scope.frameWrangler.currentFrame
        F = (abs(np.fft.fftshift(np.fft.fftn(im - im.mean()))) + 1e-2).squeeze()

        #currVoxelSizeID = self.scope.settingsDB.execute("SELECT sizeID FROM VoxelSizeHistory ORDER BY time DESC").fetchone()
        #if not currVoxelSizeID is None:
        #    voxx, voxy = self.scope.settingsDB.execute("SELECT x,y FROM VoxelSizes WHERE ID=?", currVoxelSizeID).fetchone()
            
        voxx, voxy = self.scope.getPixelSize()

        plt.figure(2)
        plt.clf()

        xd = F.shape[0]/2.
        yd = F.shape[1]/2.

        cd = xd*2*voxx/5
        #kill central spike
        F[(xd-cd):(xd+cd),(yd-cd):(yd+cd)] = F.mean()
        plt.imshow(F.T, interpolation='nearest', cmap=cm.hot)

        xd = F.shape[0]/2.
        yd = F.shape[1]/2.

        plt.plot(xd+(xd*2*voxx/1.8)*np.cos(np.arange(0, 2.1*np.pi, .1)), yd+(xd*2*voxx/1.8)*np.sin(np.arange(0, 2.1*np.pi, .1)), lw=2,label='$1.8 {\mu}m$')
        plt.plot(xd+(xd*2*voxx/2.)*np.cos(np.arange(0, 2.1*np.pi, .1)), yd+(xd*2*voxx/2.)*np.sin(np.arange(0, 2.1*np.pi, .1)), lw=1,label='$2 {\mu}m$')
        plt.plot(xd+(xd*2*voxx/1.6)*np.cos(np.arange(0, 2.1*np.pi, .1)), yd+(xd*2*voxx/1.6)*np.sin(np.arange(0, 2.1*np.pi, .1)), lw=1,label='$1.6 {\mu}m$')

        plt.xlim(xd - xd*2*voxx/1, xd + xd*2*voxx/1)
        plt.ylim(yd - xd*2*voxx/1, yd + xd*2*voxx/1)
        plt.legend()

        self.F = F


    
