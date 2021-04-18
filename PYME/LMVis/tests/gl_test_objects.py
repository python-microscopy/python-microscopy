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
import json
import threading
from collections import OrderedDict
from math import sin, cos

import numpy
from numpy import random

from PYME.Analysis.points.spherical_harmonics import cart2sph, reconstruct_from_modes
from PYME.IO.tabular import HDFSource
from PYME.simulation.wormlike2 import wormlikeChain


class TestObject(object):
    """
    This object represents a general test object. It handles the coordinates for the specific object.
    It allows translation and scaling for the object.
    It handles a list of added sub_objects. Adding sub_objects is thread safe. Changing values of the objects is not 
    secured while multi-threading.
    """
    MICROMETER_CONVERSION_CONSTANT = 1000

    def __init__(self, x, y, z, probe=0):
        """
        
        Parameters
        ----------
        x
        y
        z
        probe       either scalar value or list of scalars which matches the size of x,y and z
                        as numpy.ndarray

        """
        self._x = x
        self._y = y
        self._z = z
        self.added_objects = list()
        self.added_json = OrderedDict()
        self._probe = probe
        self._lock = threading.Lock()

    @property
    def x(self):
        new_x = self._x
        for other_object in self.added_objects:
            new_x = numpy.append(new_x, other_object.x)
        return new_x

    @property
    def y(self):
        new_y = self._y
        for other_object in self.added_objects:
            new_y = numpy.append(new_y, other_object.y)
        return new_y

    @property
    def z(self):
        new_z = self._z
        for other_object in self.added_objects:
            new_z = numpy.append(new_z, other_object.z)
        return new_z

    @property
    def probe(self):
        new_probe = None
        if self._x is not None:
            new_probe = self.add_probe(new_probe, self._probe, len(self._x))

        for other_object in self.added_objects:
            new_probe = self.add_probe(new_probe, other_object.probe)
        return new_probe

    def add_probe(self, probe_list=None, values=0, size=0):
        """
        
        Parameters
        ----------
        probe_list  list that the new values should be added to
        values      if value is scalar, use size to create a fitting list
        size        scalar value 

        Returns
        -------

        """
        if probe_list is None:
            if isinstance(values, numpy.ndarray):
                return values
            else:
                return numpy.ones((size, 1)) * self._probe
        else:
            if isinstance(values, numpy.ndarray):
                return numpy.append(probe_list, values)
            else:
                return numpy.append(probe_list, numpy.ones((size, 1)) * self._probe)

    @property
    def probe_value(self):
        return self._probe

    def translate(self, x=0, y=0, z=0):
        """
        
        Parameters
        ----------
        x       translation in x direction in micrometer
        y       translation in y direction in micrometer
        z       translation in z direction in micrometer

        Returns
        -------

        """
        self._x += x * TestObject.MICROMETER_CONVERSION_CONSTANT
        self._y += y * TestObject.MICROMETER_CONVERSION_CONSTANT
        self._z += z * TestObject.MICROMETER_CONVERSION_CONSTANT
        for other_object in self.added_objects:
            other_object.translate(x, y, z)

    def scale(self, x=1, y=1, z=1):
        self._x *= x
        self._y *= y
        self._z *= z
        for other_object in self.added_objects:
            other_object.scale(x, y, z)

    def add(self, other):
        """
        This method adds a new sub_object to this test object.
        This operation is thread safe. So many threads could try to add objects, but it won't result in a strange
        situation.
        Parameters
        ----------
        other   new sub_object

        Returns
        -------

        """
        try:
            self._lock.acquire()
            self.added_objects.append(other)
        finally:
            self._lock.release()

    def save(self, file_name):
        with open(file_name, 'wb') as csv_file:
            writer = csv.writer(csv_file)

            collection = numpy.column_stack((self.x, self.y, self.z, self.probe))
            writer.writerow(('x', 'y', 'z', 'probe'))
            writer.writerows(collection)

    def to_json(self):
        json_config = OrderedDict({'object_class': self.__class__.__name__})
        for key, value in self.added_json.items():
            json_config[key] = value
        if self.added_objects:
            json_config['objects'] = list(self.added_objects)

        return json_config

    def add_to_json(self, key, value):
        self.added_json[key] = value


class TestObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TestObject):
            return obj.to_json()
        else:
            return json.JSONEncoder.default(self, obj)


class TestObjectContainer(TestObject):
    def __init__(self):
        super(TestObjectContainer, self).__init__(None, None, None)

    @property
    def x(self):
        new_x = None
        for other_object in self.added_objects:
            if new_x is not None:
                new_x = numpy.append(new_x, other_object.x)
            else:
                new_x = other_object.x
        return new_x

    @property
    def y(self):
        new_y = None
        for other_object in self.added_objects:
            if new_y is not None:
                new_y = numpy.append(new_y, other_object.y)
            else:
                new_y = other_object.y
        return new_y

    @property
    def z(self):
        new_z = None
        for other_object in self.added_objects:
            if new_z is not None:
                new_z = numpy.append(new_z, other_object.z)
            else:
                new_z = other_object.z
        return new_z

    @property
    def probe(self):
        new_probe = None
        for other_object in self.added_objects:
            if new_probe is not None:
                new_probe = numpy.append(new_probe, other_object.probe)
            else:
                new_probe = other_object.probe
        return new_probe

    def translate(self, x=0, y=0, z=0):
        """

        Parameters
        ----------
        x       translation in x direction in micrometer
        y       translation in y direction in micrometer
        z       translation in z direction in micrometer

        Returns
        -------

        """
        for other_object in self.added_objects:
            other_object.translate(x, y, z)

    def scale(self, x=1, y=1, z=1):
        for other_object in self.added_objects:
            other_object.scale(x, y, z)


class GridContainer(TestObjectContainer):
    def __init__(self, size, offsets):
        """
        
        Parameters
        ----------
        size    (rows, columns) in nm, all > 0
        offsets (row_offset, column_offset) in nm, all > 0
        """
        self.size = size
        self.offsets = offsets
        super(TestObjectContainer, self).__init__(None, None, None)

    def shuffle(self):
        random.shuffle(self.added_objects)
        self.re_enumerate()

    def re_enumerate(self):
        object_no = 0
        for added_object in self.added_objects:
            added_object.add_to_json('row', self.get_row(object_no))
            added_object.add_to_json('column', self.get_column(object_no))
            object_no += 1

    @property
    def x(self):
        new_x = None
        item = 0
        for other_object in self.added_objects:
            added_x = numpy.copy(other_object.x)
            added_x += self.offsets[1] * self.MICROMETER_CONVERSION_CONSTANT * self.get_column(item)
            if new_x is not None:
                new_x = numpy.append(new_x, added_x)
            else:
                new_x = added_x
            item += 1
        return new_x

    @property
    def y(self):
        new_y = None
        item = 0
        for other_object in self.added_objects:
            added_y = numpy.copy(other_object.y)
            added_y -= self.offsets[0] * self.MICROMETER_CONVERSION_CONSTANT * self.get_row(item)
            if new_y is not None:
                new_y = numpy.append(new_y, added_y)
            else:
                new_y = added_y
            item += 1
        return new_y

    # @property
    # def z(self):
    #     new_z = None
    #     for other_object in self.added_objects:
    #         if new_z is not None:
    #             new_z = numpy.append(new_z, other_object.z)
    #         else:
    #             new_z = other_object.z
    #     return new_z

    def get_row(self, value):
        return value / self.size[0]

    def get_column(self, value):
        return value % self.size[0]

    def to_json(self):
        self.re_enumerate()
        json_config = super(GridContainer, self).to_json()
        json_config['size'] = self.size
        json_config['offsets'] = self.offsets
        json_config['amount_of_objects'] = len(self.added_objects)
        return json_config


class NineCollections(TestObject):

    def __init__(self):
        x = self.MICROMETER_CONVERSION_CONSTANT*((numpy.arange(270) % 27)/9 + 0.1*numpy.random.randn(270))
        y = self.MICROMETER_CONVERSION_CONSTANT*((numpy.arange(270) % 9)/3 + 0.1*numpy.random.randn(270))
        z = self.MICROMETER_CONVERSION_CONSTANT*(numpy.arange(270) % 3 + 0.1*numpy.random.randn(270))
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
    AXIS_B = 5
    AXIS_C = 2

    def __init__(self, amount_points,
                 axis_a=AXIS_A,
                 axis_b=AXIS_B,
                 axis_c=AXIS_C,
                 size=5):
        """
        
        Parameters
        ----------
        amount_points   amount of points in this ellipsoid
        axis_a          
        axis_b
        axis_c
        size            the maximum dimension of axis_a, axis_b, axis_c is max_size micrometers big
        """
        self.axis_a = axis_a
        self.axis_b = axis_b
        self.axis_c = axis_c
        self.size = size
        self.amount_of_points = amount_points
        max_axis = max(axis_a, axis_b, axis_c)
        x = size * self.MICROMETER_CONVERSION_CONSTANT * numpy.random.randn(amount_points) * axis_a / max_axis
        y = size * self.MICROMETER_CONVERSION_CONSTANT * numpy.random.randn(amount_points) * axis_b / max_axis
        z = size * self.MICROMETER_CONVERSION_CONSTANT * numpy.random.randn(amount_points) * axis_c / max_axis
        TestObject.__init__(self, x, y, z)

    def to_json(self):
        json_config = super(Ellipsoid, self).to_json()
        json_config['axis_a'] = self.axis_a
        json_config['axis_b'] = self.axis_b
        json_config['axis_c'] = self.axis_c
        json_config['size'] = self.size
        json_config['amount_of_points'] = self.amount_of_points
        return json_config


class Worm(TestObject):
    def __init__(self, kbp=100, length_per_kbp=10.0, step_length=20.0, persist_length=50, probe=0):
        chain = wormlikeChain(kbp, steplength=step_length, lengthPerKbp=length_per_kbp, persistLength=persist_length)
        self.kbp = kbp
        self.length_per_kbp = length_per_kbp
        self.step_length = step_length
        self.persist_length = persist_length

        TestObject.__init__(self, chain.xp, chain.yp, chain.zp, probe)

    def to_json(self):
        json_config = super(Worm, self).to_json()
        json_config['kbp'] = self.kbp
        json_config['length_per_kbp'] = self.length_per_kbp
        json_config['step_length'] = self.step_length
        json_config['persist_length'] = self.persist_length
        return json_config


class Vesicle(TestObject):

    WIDTH = 1

    def __init__(self, diameter=1.0, amount_points=100, hole_size=0.5 * numpy.pi, hole_pos=0):
        """

        Parameters
        ----------
        diameter        of the vesicle in micrometer
        hole_size       in rad [0-2*pi]
        hole_pos        in rad [0-2*pi]
                        0 => y=0, x=1 => right
        """

        radius = diameter * self.MICROMETER_CONVERSION_CONSTANT / 2.0
        rad = numpy.random.rand(amount_points)*(2*numpy.pi-hole_size)+hole_size/2 + hole_pos
        dist = radius - numpy.random.rand(amount_points) * self.WIDTH

        x = dist * numpy.cos(rad)
        y = dist * numpy.sin(rad)
        z = numpy.ones(x.shape) * 0.1 * numpy.random.randn(amount_points)

        self.diameter = diameter
        self.amount_of_points = amount_points
        self.hole_size = hole_size
        self.hole_pos = hole_pos
        TestObject.__init__(self, x, y, z)

    def has_hole(self):
        return self.hole_size > 0

    def to_json(self):
        json_config = super(Vesicle, self).to_json()
        json_config['diameter'] = self.diameter
        json_config['amount_of_points'] = self.amount_of_points
        json_config['has_hole'] = self.hole_size > 0
        if self.hole_size > 0:
            json_config['hole_size'] = self.hole_size
            json_config['hole_pos'] = self.hole_pos
        return json_config


class NoisePlane(TestObject):
    def __init__(self, diameter=1.0, density=20.0):
        """
        
        Parameters
        ----------
        diameter    in micrometer
        density     per micrometer^2
        """
        radius = diameter / 2.0
        amount_of_points = int(round(density * radius * radius * numpy.pi))
        radius = radius * self.MICROMETER_CONVERSION_CONSTANT
        rad = numpy.random.rand(amount_of_points) * 2 * numpy.pi
        dist = radius * numpy.sqrt(numpy.random.rand(amount_of_points))
        x = dist * numpy.cos(rad)
        y = dist * numpy.sin(rad)
        z = numpy.ones(amount_of_points) * 0.1 * numpy.random.randn(amount_of_points)
        self.diameter = diameter
        self.density = density
        TestObject.__init__(self, x, y, z)

    def to_json(self):
        json_config = super(NoisePlane, self).to_json()
        json_config['diameter'] = self.diameter
        json_config['density'] = self.density
        return json_config


class HarmonicCellBackground(TestObject):
    def __init__(self, file_name, dimensions, density=40):
        """
        
        Parameters
        ----------
        self
        file        hdf file that contains the models parameters
        dimensions  dimensions of the bounding box in micrometer
        density     density of the bounding box/the HarmonicCellBackground per micrometer^2

        Returns
        -------
        
        """
        self._dimensions = dimensions
        self._density = density
        self._file_name = file_name
        ds = HDFSource(file_name, tablename='Data')

        self._center = ds['centre']
        self._modes = zip(ds['m_modes'], ds['n_modes'])
        self._z_scale = float(ds['z_scale'][0])
        self._coefficients = ds['coefficients']

        # amount of points that is created for monte carlo
        amount_of_points = density * dimensions[0] * dimensions[1] * dimensions[2]
        # create bounding box, that is filled with points for monte carlo
        x = (numpy.random.rand(amount_of_points) - 0.5) * self.MICROMETER_CONVERSION_CONSTANT * dimensions[0]
        y = (numpy.random.rand(amount_of_points) - 0.5) * self.MICROMETER_CONVERSION_CONSTANT * dimensions[1]
        z = (numpy.random.rand(amount_of_points) - 0.5) * self.MICROMETER_CONVERSION_CONSTANT * dimensions[2]
        z *= self._z_scale  # spherical harmonics work better with bigger z values

        generated_theta, generated_phi, generated_r = cart2sph(x, y, z)

        r_spherical_harmonics = map(lambda theta, phi: self.get_radius(theta, phi), generated_theta, generated_phi)

        mask = generated_r < r_spherical_harmonics

        x = x[mask]
        y = y[mask]
        z = z[mask] / self._z_scale  # restore real z_values
        TestObject.__init__(self, x, y, z)

    def to_json(self):
        json_config = super(HarmonicCellBackground, self).to_json()
        json_config['file_name'] = self._file_name
        json_config['density'] = self._density
        json_config['dimensions'] = self._dimensions
        return json_config

    def get_radius(self, theta, phi):
        return reconstruct_from_modes(self._modes, self._coefficients, theta, phi)

    def get_coordinates(self, theta, phi, radius):
        """
        this method will create a point within the surface in coordinates
        Parameters
        ----------
        theta   [0, pi[
        phi     [0, 2pi[
        radius  [0,1[

        Returns
        -------
        (x, y, z)   in micrometers
    
        """
        real_radius = self.get_radius(theta, phi) / TestObject.MICROMETER_CONVERSION_CONSTANT
        x = radius * real_radius * sin(theta) * cos(phi)
        y = radius * real_radius * sin(theta) * sin(phi)
        z = radius * real_radius * cos(theta) / self._z_scale

        return x, y, z


class HarmonicCell(TestObjectContainer):
    def __init__(self, input_file, dimensions):

        TestObjectContainer.__init__(self)
        test_object = TestObjectContainer()
        test_harmonic = HarmonicCellBackground(input_file, dimensions, 40)
        test_object.add(test_harmonic)

        chromosome = 0
        amount_chromosomes = random.randint(0, 5)
        while chromosome < amount_chromosomes:
            worm = Worm(250, probe=chromosome+1)
            theta = random.rand() * numpy.pi
            phi = 2 * random.rand() * numpy.pi
            radius = 0.8 * random.rand()
            x, y, z = test_harmonic.get_coordinates(theta, phi, radius)
            worm.translate(x, y, z)
            test_object.add(worm)
            chromosome += 1
        test_object = ExponentialClusterizer(test_object, 4, 15)
        self.add(test_object)


class Clusterizer(TestObject):
    def __init__(self, test_object, multiply, distance):
        self.test_object = test_object
        self.multiply = multiply
        self.distance = distance

        base_points_x, base_points_y, base_points_z, self.amount_of_points, probes = self.get_points()
        offsets = self.get_offsets()

        new_positions_x = base_points_x + offsets[0]
        new_positions_y = base_points_y + offsets[1]
        new_positions_z = base_points_z + offsets[2]

        TestObject.__init__(self, new_positions_x, new_positions_y, new_positions_z, probe=probes)

    def get_points(self):
        test_object = self.test_object
        return (numpy.repeat(test_object.x, self.multiply),
                numpy.repeat(test_object.y, self.multiply),
                numpy.repeat(test_object.z, self.multiply),
                len(test_object.x)*self.multiply,
                numpy.repeat(test_object.probe, self.multiply))

    def get_offsets(self):
        offsets_x = (numpy.random.randn(self.amount_of_points) - 0.5) * 2 * self.distance
        offsets_y = (numpy.random.randn(self.amount_of_points) - 0.5) * 2 * self.distance
        offsets_z = (numpy.random.randn(self.amount_of_points) - 0.5) * 2 * self.distance
        return offsets_x, offsets_y, offsets_z

    def to_json(self):
        json_config = super(Clusterizer, self).to_json()
        json_config['multiply'] = self.multiply
        json_config['distance'] = self.distance
        json_config['objects'] = self.test_object.to_json()
        return json_config


class ExponentialClusterizer(Clusterizer):
    def __init__(self, test_object, expectation_value, distance):
        self.expectation_value = expectation_value
        super(ExponentialClusterizer, self).__init__(test_object, 0, distance)

    def get_points(self):
        amount_of_input_points = len(self.test_object.x)
        cluster_sizes = numpy.round(
            numpy.random.exponential(self.expectation_value, amount_of_input_points)).astype(int)
        base_points_x = numpy.repeat(self.test_object.x, cluster_sizes)
        base_points_y = numpy.repeat(self.test_object.y, cluster_sizes)
        base_points_z = numpy.repeat(self.test_object.z, cluster_sizes)
        probes = numpy.repeat(self.test_object.probe, cluster_sizes)
        amount_of_points = len(base_points_x)
        return base_points_x, base_points_y, base_points_z, amount_of_points, probes

    def to_json(self):
        json_config = super(ExponentialClusterizer, self).to_json()
        json_config['expectation_value'] = self.expectation_value
        json_config['distance'] = self.distance
        json_config['amount_of_points'] = len(self.x)

        return json_config
