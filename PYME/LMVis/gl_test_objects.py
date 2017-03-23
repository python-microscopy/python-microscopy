#!/usr/bin/python

# gl_test_objects.py
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
import numpy


class TestObject(object):
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def translate(self, x=0, y=0, z=0):
        self._x += x
        self._y += y
        self._z += z


class NineCollections(TestObject):

    def __init__(self):
        x = 5e3*((numpy.arange(270) % 27)/9 + 0.1*numpy.random.randn(270))
        y = 5e3*((numpy.arange(270) % 9)/3 + 0.1*numpy.random.randn(270))
        z = 5e3*(numpy.arange(270) % 3 + 0.1*numpy.random.randn(270))
        TestObject.__init__(self, x, y, z)


class Cloud(TestObject):
    DISTANCE = 200.0

    def __init__(self, amount_points):
        x = Cloud.DISTANCE * numpy.random.randn(amount_points)
        y = Cloud.DISTANCE * numpy.random.randn(amount_points)
        z = Cloud.DISTANCE * numpy.random.randn(amount_points)
        TestObject.__init__(self, x, y, z)
