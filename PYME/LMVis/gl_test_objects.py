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
import csv

import numpy

from PYME.Acquire.Hardware.Simulator.wormlike2 import wormlikeChain


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

    def scale(self, x=1, y=1, z=1):
        self._x *= x
        self._y *= y
        self._z *= z

    def __add__(self, other):
        new_x = numpy.append(other.x, self._x)
        new_y = numpy.append(other.y, self._y)
        new_z = numpy.append(other.z, self._z)
        return TestObject(new_x,new_y,new_z)

    def save(self, file_name):
        with open(file_name, 'wb') as csv_file:
            # fieldnames = ['x', 'z', 'z']
            writer = csv.writer(csv_file)

            # writer.writeheader()
            collection = numpy.column_stack((self._x, self._y, self._z))
            writer.writerow(('x', 'y', 'z'))
            writer.writerows(collection)

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


class Ellipsoid(TestObject):

    AXIS_A = 4
    AXIS_B = 8
    AXIS_C = 2

    def __init__(self, amount_points,
                 axis_a=AXIS_A,
                 axis_b=AXIS_B,
                 axis_c=AXIS_C):
        max_axis = max(axis_a, axis_b, axis_c)
        x = 5e3*numpy.random.randn(amount_points) * axis_a / max_axis
        y = 5e3*numpy.random.randn(amount_points) * axis_b / max_axis
        z = 5e3*numpy.random.randn(amount_points) * axis_c / max_axis
        TestObject.__init__(self, x, y, z)


class Worm(TestObject):
    def __init__(self, kbp=200):
        chain = wormlikeChain(kbp, steplength=50)
        TestObject.__init__(self, chain.xp, chain.yp, chain.zp)


class Ring(TestObject):

    DIAMETER = 5e3
    WIDTH = 1000

    def __init__(self, amount_points, hole_size=0.4, hole_pos=0):
        """

        Parameters
        ----------
        amount_points
        hole_size       in rad [0-2*pi]
        hole_pos        in rad [0-2*pi]
                        0 => y=0, x=1 => right
        """
        rad = numpy.random.rand(amount_points)*(2*numpy.pi-hole_size)+hole_size/2 + hole_pos
        dist = self.DIAMETER - numpy.random.rand(amount_points) * self.WIDTH

        x = dist * numpy.cos(rad)
        y = dist * numpy.sin(rad)
        z = numpy.ones(x.shape) * 0.1*numpy.random.randn(amount_points)

        TestObject.__init__(self, x, y, z)





