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
import sys
from wx import wx
import numpy as np

from PYME.LMVis import gl_test_objects
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


class Fish(TestApp):
    def __init__(self, *args):
        self.to = Ellipsoid(4000)
        concentration = Worm(100)
        concentration.translate(1000, 0, 0)
        self.to += concentration
        concentration = Worm(100)
        concentration.translate(-2300, 500, 0)
        self.to += concentration
        super(Fish, self).__init__(*args)

    def OnInit(self):
        self.setup()

        self._canvas.pointSize = 50

        self._canvas.setPoints3D(self.to.x, self.to.y, self.to.z, normalize(self.to.z),
                                 self._canvas.cmap, self._canvas.clim, mode='pointsprites')
        self._canvas.recenter(self.to.x, self.to.y)

        self.done()
        return True

    def save(self, file_name):
        self.to.save(file_name)


class Rings(TestApp):
    def __init__(self, *args):
        self.to = Ring(2000)

        'first step'
        offset = -10000
        scale = 0.5
        new_ring = Ring(1000, hole_pos=numpy.pi / 2)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(-5000, offset, 0)

        self.to += new_ring

        new_ring = Ring(1000, hole_pos=0.75 * numpy.pi / 2)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(5000, offset, 0)

        self.to += new_ring

        'second step'
        offset = -15000
        scale = 0.25
        new_ring = Ring(250, hole_pos=3 * numpy.pi / 2)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(0, offset, 0)

        self.to += new_ring

        new_ring = Ring(250, hole_pos=2 * numpy.pi)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(7000, offset, 0)

        self.to += new_ring

        new_ring = Ring(250, hole_size=0, hole_pos=4 * numpy.pi / 2)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(-7000, offset, 0)

        self.to += new_ring

        'third step'
        offset = -18000
        scale = 0.125
        new_ring = Ring(100, hole_pos=7 * numpy.pi / 4)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(-4000, offset, 0)

        self.to += new_ring

        new_ring = Ring(100, hole_size=0, hole_pos=7 * numpy.pi / 4)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(4000, offset, 0)

        self.to += new_ring

        'fourth step'
        offset = -20000
        scale = 0.08
        new_ring = Ring(75, hole_pos=5 * numpy.pi / 4)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(-4000, offset, 0)

        self.to += new_ring

        new_ring = Ring(75, hole_size=0, hole_pos=7 * numpy.pi / 4)
        new_ring.scale(scale, scale, scale)
        new_ring.translate(4000, offset, 0)

        self.to += new_ring

        super(Rings, self).__init__(*args)

    def OnInit(self):
        self.setup()

        self._canvas.pointSize = 50

        self._canvas.setPoints3D(self.to.x, self.to.y, self.to.z, normalize(self.to.z),
                                 self._canvas.cmap, self._canvas.clim, mode='pointsprites')
        self._canvas.recenter(self.to.x, self.to.y)

        self.done()
        return True

    def save(self, file_name):
        self.to.save(file_name)


class Worms(TestApp):
    def __init__(self, *args):
        self.to = gl_test_objects.Worm()
        super(Worm, self).__init__(*args)

    def OnInit(self):
        self.setup()

        self._canvas.pointSize = 50

        self._canvas.setPoints3D(self.to.x, self.to.y, self.to.z, normalize(self.to.z),
                                 self._canvas.cmap, self._canvas.clim, mode='pointsprites')
        self._canvas.recenter(self.to.x, self.to.y)

        self.done()
        return True

    def save(self, file_name):
        self.to.save(file_name)


def normalize(values):
    return (values - min(values)) / (max(values) - min(values))


def main():
    app = Fish()
    app.save(sys.argv[1])
    app.MainLoop()


if __name__ == '__main__':
    main()
