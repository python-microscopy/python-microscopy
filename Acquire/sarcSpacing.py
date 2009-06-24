from pylab import *
from PYME import cSMI
import wx


class SarcomereChecker:
    def __init__(self, parent, menu, scope, key = 'F12'):
        self.scope = scope

        idSarcCheck = wx.NewId()

        self.menu = wx.Menu(title = '')

        self.menu.Append(idSarcCheck, 'Check Sarcomere Spacing\t%s' % key)
        wx.EVT_MENU(parent, idSarcCheck, self.OnCheckSpacing)


        menu.Append(menu=self.menu, title = '&Utils')
        self.mbar = menu
        self.mpos = menu.GetMenuCount() - 1


    def OnCheckSpacing(self,event):
        voxx = 0.07
        voxy = 0.07
        
        im = cSMI.CDataStack_AsArray(self.scope.pa.ds, 0)
        F = (abs(fftshift(fftn(im - im.mean()))) + 1e-2).squeeze()

        currVoxelSizeID = self.scope.settingsDB.execute("SELECT sizeID FROM VoxelSizeHistory ORDER BY time DESC").fetchone()
        if not currVoxelSizeID == None:
            voxx, voxy = self.scope.settingsDB.execute("SELECT x,y FROM VoxelSizes WHERE ID=?", currVoxelSizeID).fetchone()

        figure(2)
        clf()

        xd = F.shape[0]/2.
        yd = F.shape[1]/2.

        cd = xd*2*voxx/5
        #kill central spike
        F[(xd-cd):(xd+cd),(yd-cd):(yd+cd)] = F.mean()
        imshow(F.T, interpolation='nearest', cmap=cm.hot)

        xd = F.shape[0]/2.
        yd = F.shape[1]/2.

        plot(xd+(xd*2*voxx/1.8)*cos(arange(0, 2.1*pi, .1)), yd+(xd*2*voxx/1.8)*sin(arange(0, 2.1*pi, .1)), lw=2,label='$1.8 {\mu}m$')
        plot(xd+(xd*2*voxx/2.)*cos(arange(0, 2.1*pi, .1)), yd+(xd*2*voxx/2.)*sin(arange(0, 2.1*pi, .1)), lw=1,label='$2 {\mu}m$')
        plot(xd+(xd*2*voxx/1.6)*cos(arange(0, 2.1*pi, .1)), yd+(xd*2*voxx/1.6)*sin(arange(0, 2.1*pi, .1)), lw=1,label='$1.6 {\mu}m$')

        xlim(xd - xd*2*voxx/1, xd + xd*2*voxx/1)
        ylim(yd - xd*2*voxx/1, yd + xd*2*voxx/1)
        legend()

        self.F = F


    
