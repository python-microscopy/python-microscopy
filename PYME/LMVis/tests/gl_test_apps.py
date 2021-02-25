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
import argparse
import time
from ast import literal_eval
from math import floor

# import pylab
import matplotlib.cm
import datetime
from wx import wx

from PYME.LMVis.gl_render3D_shaders import LMGLShaderCanvas
from PYME.LMVis.tests import gl_test_objects
from PYME.LMVis.tests.gl_test_objects import *


class TestApp(wx.App):
    def __init__(self, *args):
        wx.App.__init__(self)
        self._frame = None
        self._canvas = None

    def OnInit(self):
        self.setup()
        self._canvas.displayMode = '3D'
        self._canvas.pointSize = 10

        self._canvas.setPoints3D(self.to.x, self.to.y, self.to.z, normalize(self.to.z),
                                 self._canvas.cmap, self._canvas.clim, mode='pointsprites',
                                 )
        self._canvas.recenter(self.to.x, self.to.y)
        self.done()
        return True

    def setup(self):
        self._frame = wx.Frame(None, -1, 'ball_wx', wx.DefaultPosition, wx.Size(800, 800))
        self._canvas = LMGLShaderCanvas(self._frame)
        self._canvas.gl_context.SetCurrent(self._canvas)
        self._canvas.SetCurrent()
        self._canvas.initialize()
        self._canvas.setCMap(matplotlib.cm.hot)
        self._canvas.clim = [0.1, 0.9]
        self._canvas.displayMode = '3D'
        self._canvas.pointSize = 50

    def done(self):
        self._canvas.Refresh()
        self._frame.Show()
        self.SetTopWindow(self._frame)

    def save(self, test_object_file, configuration_file):
        if self.to:
            self.to.save(test_object_file)
            self.to.add_to_json('file', test_object_file)
            self.to.add_to_json('time', '%s' % datetime.datetime.now())
            configurations = json.dumps(self.to, cls=TestObjectEncoder, indent=4)
            with open(configuration_file, 'wb') as f:
                f.writelines(configurations)


class XTestApp(TestApp):
    def OnInit(self):
        self.setup()
        to = Cloud(40)
        self._canvas.setPoints3D(to.x, to.y, to.z, mode='pointsprites')
        self._canvas.recenter(to.x, to.y)
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
        self.to = Ellipsoid(3000)
        self.to = ExponentialClusterizer(self.to, 4, 30)
        concentration = Worm(250)
        concentration.translate(1, 0, 0)
        self.to.add(concentration)
        concentration = Worm(200)
        concentration.translate(-2.3, 0.5, 0)
        self.to.add(concentration)
        super(Fish, self).__init__(*args)


class Vesicles(TestApp):
    def __init__(self, args):
        # scales in micrometer
        scales = [0.125, 0.100, 0.080]
        rows = columns = 11
        row_shift = - 0.6  # row is 'down', so negative
        column_shift = 0.6
        total_dict = []

        # row_shift = - 0.2  # row is 'down', so negative
        # column_shift = 0.2
        base_amount_of_points = 35
        minimum_amount_of_points = 10
        self.to = TestObjectContainer()
        for row in numpy.arange(1, rows):
            row_container = TestObjectContainer()
            row_container.add_to_json('row', row)
            row_list = []
            for column in numpy.arange(1, columns):
                hole_position = numpy.random.random()
                hole_position_with_pi = hole_position * 2 * numpy.pi
                has_hole = numpy.random.random() >= 0.5
                scale = scales[random.randint(0, len(scales))]
                amount_points = max(int(round(numpy.random.normal(base_amount_of_points, 20))),
                                    minimum_amount_of_points)

                if has_hole:
                    new_test_object = Vesicle(diameter=scale, amount_points=amount_points,
                                              hole_pos=hole_position_with_pi)
                    row_list.append(1)
                else:
                    new_test_object = Vesicle(diameter=scale, amount_points=amount_points, hole_size=0)
                    row_list.append(0)

                row_shift_extra = 0.3 * floor((row - 1) / 2)
                new_test_object.translate(column_shift * column, row_shift * row - row_shift_extra, 0)

                jitter_factor = 0.05
                jitter_x = numpy.random.rand() * jitter_factor
                jitter_y = numpy.random.rand() * jitter_factor
                new_test_object.translate(jitter_x, jitter_y, 0)
                row_container.add(new_test_object)
                new_test_object.add_to_json('column', column)

            self.to.add(row_container)
            total_dict.append(row_list)

        noise = NoisePlane(20, 20)
        noise.translate(column_shift * column / 2, row_shift * row / 2, 0)
        self.to.add(noise)

        self.to = ExponentialClusterizer(self.to, 4, 10)
        if args.result_table is not None:
            self.save_result_csv(total_dict, args.result_table)

        super(Vesicles, self).__init__(args)

    @staticmethod
    def save_result_csv(result_dict, file_name):
        with open(file_name, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(result_dict)


class Worms(TestApp):
    def __init__(self, *args):
        self.to = TestObjectContainer()
        self.to.add(ExponentialClusterizer(gl_test_objects.Worm(250, probe=1), 2, 0))
        self.to.add(ExponentialClusterizer(gl_test_objects.Worm(250, probe=2), 2, 0))
        self.to = ExponentialClusterizer(self.to, 2, 1)
        super(Worms, self).__init__(*args)


class HarmonicCells(TestApp):
    def __init__(self, args):
        self._input_file = args.harmonics_file
        self._dimensions = literal_eval(args.harmonics_dimensions)
        self.to = TestObjectContainer()
        row_offset = - self._dimensions[1] * 2.2
        column_offset = self._dimensions[0] * 1.2
        for row in numpy.arange(4):
            for column in numpy.arange(4):
                print('Generating: row {}: column {}'.format(row, column))
                cell = self.create_harmonic_cell()
                cell.translate(column * column_offset, row * row_offset, 0)
                cell.add_to_json('row', row + 1)
                cell.add_to_json('column', column + 1)
                self.to.add(cell)
        super(HarmonicCells, self).__init__(args)

    def create_harmonic_cell(self):
        return HarmonicCell(self._input_file, self._dimensions)


class GridTestApp(TestApp):

    def __init__(self, grid_container, args):
        self.modes = args.modes
        self._grid_container = grid_container
        super(GridTestApp, self).__init__(args)

    def save(self, output_csv, output_json):
        """
        
        Parameters
        ----------
        output_csv      the csv that contains all points
        output_json     the json that contains the configuration of the points

        Returns
        -------

        """
        for i in self.modes:
            super(GridTestApp, self).save(output_csv.format(i), output_json.format(i))
            self.shuffle_object.shuffle()


class GridHarmonicCells(GridTestApp):
    def __init__(self, args):
        self._input_file = args.harmonics_file
        self._dimensions = literal_eval(args.harmonics_dimensions)
        rows = columns = 4
        row_offset = self._dimensions[1] * 1.2
        column_offset = self._dimensions[0] * 1.2
        grid_container = GridContainer((rows, columns), (row_offset, column_offset))
        self.to = grid_container
        self.shuffle_object = self.to
        for row in numpy.arange(rows):
            for column in numpy.arange(columns):
                print('Generating: row {}: column {}'.format(row, column))
                cell = self.create_harmonic_cell()
                self.to.add(cell)
        super(GridHarmonicCells, self).__init__(grid_container, args)

    def create_harmonic_cell(self):
        return HarmonicCell(self._input_file, self._dimensions)


class GridVesicles(GridTestApp):
    def __init__(self, args):
        # scales in micrometer
        scales = [0.125, 0.100, 0.080]
        rows = columns = 10
        row_shift = 0.6
        column_shift = 0.6
        total_dict = []

        # row_shift = - 0.2  # row is 'down', so negative
        # column_shift = 0.2
        base_amount_of_points = 35
        variance = 20
        minimum_amount_of_points = 10
        grid_container = GridContainer((rows, columns), (row_shift, column_shift))
        self.shuffle_object = grid_container
        for row in numpy.arange(0, rows):
            row_list = []
            for column in numpy.arange(0, columns):
                hole_position = numpy.random.random()
                hole_position_with_pi = hole_position * 2 * numpy.pi
                has_hole = numpy.random.random() >= 0.5
                scale = scales[random.randint(0, len(scales))]
                amount_points = max(int(round(numpy.random.normal(base_amount_of_points, variance))),
                                    minimum_amount_of_points)

                if has_hole:
                    new_test_object = Vesicle(diameter=scale, amount_points=amount_points,
                                              hole_pos=hole_position_with_pi)
                    row_list.append(1)
                else:
                    new_test_object = Vesicle(diameter=scale, amount_points=amount_points, hole_size=0)
                    row_list.append(0)

                jitter_factor = 0.05
                jitter_x = numpy.random.rand() * jitter_factor
                jitter_y = numpy.random.rand() * jitter_factor
                new_test_object.translate(jitter_x, jitter_y, 0)
                grid_container.add(ExponentialClusterizer(new_test_object, 4, 10))

            total_dict.append(row_list)

        noise = NoisePlane(20, 20)
        noise.translate(column_shift * column / 2, row_shift * row / 2, 0)
        self.to = TestObjectContainer()
        self.to.add(grid_container)
        self.to.add(ExponentialClusterizer(noise, 4, 10))

        super(GridVesicles, self).__init__(grid_container, args)


def normalize(values):
    return (values - min(values)) / (max(values) - min(values))


def main():
    args = parse()
    start_time = time.time()
    app = eval(args.testapp)(args)
    app.save(args.output_csv, args.output_json)
    end_time = time.time()
    print('Duration to generate the test objects in s: {}'.format(end_time-start_time))
    app.MainLoop()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('testapp', help="testapp that should be used", default=Vesicles)
    parser.add_argument('output_csv', help='file that should be used to save the generated points as csv')
    parser.add_argument('output_json', help='file that should be used to save the configuration as json')
    parser.add_argument('--harmonics_file', help='the file used for input spherical harmonics data')
    parser.add_argument('--harmonics_dimensions', help='the dimensions of the bounding box in micrometer (x, y, z)')
    parser.add_argument('--modes', help='the modes of samples for grids', nargs='+')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
