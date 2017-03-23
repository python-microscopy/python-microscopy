#!/usr/bin/python

# gl_test_apps.py
#
# Copyright Michael Graff
#   graff@hm.edu
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
import pylab
from wx import wx
import numpy as np

from PYME.LMVis.gl_render3D_shaders import LMGLShaderCanvas
from PYME.LMVis.gl_test_objects import *


class TestApp(wx.App):
    def __init__(self, *args):
        wx.App.__init__(self, *args)

    def OnInit(self):
        self.setup()
        to = NineCollections()
        self._canvas.displayMode = '3D'
        self._canvas.pointSize = 50

        self._canvas.setPoints3D(to.x, to.y, to.z, mode='pointsprites')
        self._canvas.recenter(to.x, to.y)
        self.done()
        return True

    def setup(self):
        self._frame = wx.Frame(None, -1, 'ball_wx', wx.DefaultPosition, wx.Size(800, 800))
        self._canvas = LMGLShaderCanvas(self._frame)
        self._canvas.gl_context.SetCurrent(self._canvas)
        self._canvas.SetCurrent()
        self._canvas.initialize()
        self._canvas.setCMap(pylab.cm.hot)
        self._canvas.clim = [0, 1]
        self._canvas.displayMode = '3D'
        self._canvas.pointSize = 10

    def done(self):
        self._canvas.Refresh()
        self._frame.Show()
        self.SetTopWindow(self._frame)


class XTestApp(TestApp):
    def OnInit(self):
        self.setup()
        to = Cloud(40)
        self._canvas.setPoints3D(to.x, to.y, to.z, mode='pointsprites')
        self._canvas.recenter(to.x, to.y)
        self.done()
        return True


class TwtyToFotyTestApp(TestApp):
    def OnInit(self):
        self.setup()
        to = Cloud(100)
        to2 = Cloud(200)
        to2.translate(2000)
        self._canvas.setPoints3D(to.x, to.y, to.z, normalize(to.z), self._canvas.cmap, self._canvas.clim,
                                 mode='pointsprites')

        self._canvas.setPoints3D(to2.x, to2.y, to2.z, normalize(to2.z), self._canvas.cmap, self._canvas.clim,
                                 mode='pointsprites')
        self._canvas.recenter(np.append(to.x, to2.x), np.append(to.y, to2.y))
        self.done()
        return True


class MassTest(TestApp):
    def OnInit(self):
        self.setup()
        to = Cloud(1000000)
        self._canvas.setPoints3D(to.x, to.y, to.z, normalize(to.z), self._canvas.cmap, self._canvas.clim,
                                 mode='pointsprites')
        self._canvas.recenter(to.x, to.y)
        self.done()
        return True


def normalize(values):
    return (values - min(values)) / (max(values) - min(values))


def main():
    app = MassTest()
    app.MainLoop()


if __name__ == '__main__':
    main()
