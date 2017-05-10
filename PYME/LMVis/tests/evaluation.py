#!/usr/bin/python

# evaluation.py
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
import csv
import json
from abc import abstractmethod
import matplotlib.pyplot as plt
from distutils.util import strtobool

import numpy
import pylab

TOTAL_BY_SIZES = 'Total numbers grouped by sizes'

RATIO_BY_SIZES = 'Success ratio grouped by sizes'

RATIO_BY_AMOUNT = 'Success ratio grouped by amount of particles'


class Evaluation(object):
    def __init__(self, guess_table, json_file_name):
        """
        
        Parameters
        ----------
        guess_table     contains only the answers of the user (no headlines etc.)
        json_file_name  contains the configuration json
        """
        self._guess_table_name = guess_table
        self._guess_table = list()
        self._json_file = None

        with open(json_file_name, 'r') as f:
            self._json_file = json.load(f)

        with open(guess_table, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self._guess_table.append(row)

    def print_table(self):
        print(self._guess_table)

    def get_guess_size(self):
        """
        
        Returns
        -------
        (int, int)  (rows, columns) in the table     
        """
        return len(self._guess_table), len(self._guess_table[0])

    def get_guess_at(self, row, column):
        return self._guess_table[row][column]

    def get_object_at(self, row, column):
        grid = self.find_grid()
        grid_size = self.get_grid_size(grid)
        offset = row * grid_size[0] + column
        return grid['objects'][offset]

    def find_grid(self):
        return Evaluation.locate_class(self._json_file, "GridContainer")

    def get_grid_size(self, grid=None):
        if grid is None:
            return self.find_grid()['size']
        else:
            return grid['size']

    def get_name(self):
        return self._guess_table_name

    @staticmethod
    def locate_class(item, class_name):
        """
        This method locates the closes representation of a object of the given class
        Parameters
        ----------
        item        item, that should be searched
        class_name  like given in the json for the field 'object_class'

        Returns
        -------

        """

        if item['object_class'] == class_name:
            return item

        next_test_object = item['objects']
        if isinstance(next_test_object, list):
            for child in next_test_object:
                result = Evaluation.locate_class(child, class_name)
                if result is not None:
                    return result
        elif isinstance(next_test_object, dict):
                return Evaluation.locate_class(next_test_object, class_name)
        else:
            return None


class EvaluationDecorator(object):
    def __init__(self, evaluation):
        self._evaluation = evaluation

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def print_statistic(self):
        pass


class Plot(object):

    created_plots = {}

    LIST_NAME = 'list_name'
    KEYS = 'keys'
    VALUES = 'values'

    def __init__(self, figure_name):
        self._figure_name = figure_name
        plt.figure(self.get_figure_name())
        plt.title(figure_name)
        self._lists = list()

    def get_figure_name(self):
        return self._figure_name

    def set_y_axes(self, limits):
        self.set_active_figure()
        axes = plt.gca()
        axes.set_ylim(limits)

    def set_active_figure(self):
        plt.figure(self.get_figure_name())

    @abstractmethod
    def draw(self):
        pass

    @staticmethod
    def add_created_plot(key, plot):
        Plot.created_plots[key] = plot

    @staticmethod
    def get_created_plot(key):
        return Plot.created_plots[key]


class ListPlot(Plot):

    def __init__(self, figure_name):
        super(ListPlot, self).__init__(figure_name)

    def add_list(self, keys, values, legend_name):
        self._lists.append({Plot.LIST_NAME: legend_name, Plot.KEYS: keys, Plot.VALUES: values})

    def draw(self):
        self.set_active_figure()
        for entry in self._lists:
            name = self.get_figure_name()

            if 'sprites' in name:
                plt.plot(entry[Plot.KEYS], entry[Plot.VALUES], label=entry[Plot.LIST_NAME], marker='o')
            elif 'point' in name:
                plt.plot(entry[Plot.KEYS], entry[Plot.VALUES], label=entry[Plot.LIST_NAME], marker='.')
            elif 'triang' in name:
                plt.plot(entry[Plot.KEYS], entry[Plot.VALUES], label=entry[Plot.LIST_NAME], marker='^')
            else:
                plt.plot(entry[Plot.KEYS], entry[Plot.VALUES], label=entry[Plot.LIST_NAME])

        plt.grid(True)
        plt.legend(loc=2, prop={'size': 6})
        plt.draw()


class BarPlot(Plot):

    LIST_NAMES = 'LIST_NAMES'

    def __init__(self, figure_name):
        super(BarPlot, self).__init__(figure_name)
        self._amount_of_classes = 0
        self._bar_width = 1

    def add_bars(self, keys, lists, legend_name, list_names):
        self._lists.append({Plot.LIST_NAME: legend_name, Plot.KEYS: keys, Plot.VALUES: lists,
                            BarPlot.LIST_NAMES: list_names})
        self._amount_of_classes += 1

    def set_bar_width(self, value):
        self._bar_width = value

    def draw(self):
        self.set_active_figure()
        index = 0
        width = self._bar_width / self._amount_of_classes
        color_map = pylab.cm.hsv
        bottom = None
        for entry in self._lists:
            color = color_map(index*1.0/self._amount_of_classes)[:3]
            color2 = color_map(index*1.0/self._amount_of_classes+0.05)[:3]
            centered_keys = [round(float(x) - self._bar_width / 2.0 + index * width, 4) for x in entry[Plot.KEYS]]
            print(centered_keys)
            sub_index = 0
            for specific_list in entry[Plot.VALUES]:
                use_color = color
                if sub_index % 2 == 0:
                    use_color = color2
                label = '{}_{}'.format(entry[Plot.LIST_NAME], entry[BarPlot.LIST_NAMES][sub_index])
                plt.bar(centered_keys, specific_list, width=width, label=label,
                        color=use_color, bottom=bottom)
                bottom = specific_list
                sub_index += 1
            index += 1
            bottom = None

        plt.grid(True)
        plt.legend(loc=2, prop={'size': 6})
        plt.draw()


class VesicleEvaluation(EvaluationDecorator):

    def __init__(self, evaluation):
        super(VesicleEvaluation, self).__init__(evaluation)
        self._correct_guesses = list()
        self._wrong_guesses = list()

    def evaluate(self):

        size = self._evaluation.get_guess_size()
        for row in numpy.arange(0, size[0]):
            for column in numpy.arange(0, size[1]):
                #  get the answer of the user
                guess = self._evaluation.get_guess_at(row, column)

                #  get the correct answer according to the json
                obj = Evaluation.locate_class(self._evaluation.get_object_at(row, column), 'Vesicle')
                value = obj['has_hole']

                #  compare the results
                same = strtobool(guess.lower()) == bool(value)

                if same:
                    self._correct_guesses.append(obj)
                else:
                    self._wrong_guesses.append(obj)

    def print_statistic(self):
        ratio = len(self._correct_guesses) * 1.0 / (len(self._correct_guesses) + len(self._wrong_guesses))
        print('ratio: {}%'.format(ratio*100))
        self.get_ratio_for_sizes()
        self.get_ratio_for_amounts()

    def get_ratio_for_sizes(self):
        result_lists = {'good': self._correct_guesses, 'wrong': self._wrong_guesses}

        # sums = {size1:{correct:X, wrong:Y}, size 2:{}...}
        sums = {}
        #  result_list {good, wrong}
        for result_list in result_lists:

            for guess in result_lists[result_list]:
                size = guess['diameter']
                if str(size) in sums:
                    sub_dict = sums[str(size)]
                    if result_list in sub_dict:
                        sub_dict[result_list] = sub_dict[result_list] + 1
                    else:
                        sub_dict[result_list] = 1
                else:
                    sums[str(size)] = {result_list: 1}

        ratios = list()
        keys = list()
        lists = [list(), list()]
        for size in sorted(sums):
            try:
                absolute_good = sums[size]['good']
            except KeyError:
                absolute_good = 0
            lists[0].append(absolute_good)
            try:
                absolute_wrong = sums[size]['wrong']
            except KeyError:
                absolute_wrong = 0
            lists[1].append(absolute_wrong)
            absolute_total = absolute_good + absolute_wrong
            ratio = absolute_good * 100.0 / absolute_total
            ratios.append(ratio)
            keys.append(size)
            print('ratio size {}: {}% ({}/{})'.format(
                size,
                ratio, absolute_good, absolute_total))

        Plot.get_created_plot(RATIO_BY_SIZES).add_list(keys, ratios, self._evaluation.get_name())

        Plot.get_created_plot(TOTAL_BY_SIZES).add_bars(keys, lists, self._evaluation.get_name(), ['good', 'wrong'])
        # self.bar_lists(keys, lists, ['good', 'wrong'], TOTAL_BY_SIZES,
        #                y_axis=[0, 100], y_ticks=numpy.arange(0, 5, 100))

    def get_ratio_for_amounts(self):
        result_lists = {'good': self._correct_guesses, 'wrong': self._wrong_guesses}

        # sums = {size1:{correct:X, wrong:Y}, size 2:{}...}
        sums = {}
        step_size = 10
        #  result_list {good, wrong}
        for result_list in result_lists:

            for guess in result_lists[result_list]:
                real_size = guess['amount_of_points']

                size = real_size - real_size % step_size
                if str(size) in sums:
                    sub_dict = sums[str(size)]
                    if result_list in sub_dict:
                        sub_dict[result_list] = sub_dict[result_list] + 1
                    else:
                        sub_dict[result_list] = 1
                else:
                    sums[str(size)] = {result_list: 1}

        ratios = list()
        keys = list()
        for size in sorted(sums):
            absolute_good = 0
            absolute_wrong = 0
            if 'good' in sums[size]:
                absolute_good = sums[size]['good']
            if 'wrong' in sums[size]:
                absolute_wrong = sums[size]['wrong']
            absolute_total = absolute_good + absolute_wrong
            ratio = absolute_good * 100.0 / absolute_total
            ratios.append(ratio)
            keys.append(size)
            print('ratio amount of points {}: {}% ({}/{})'.format(
                size,
                ratio, absolute_good, absolute_total))

        Plot.get_created_plot(RATIO_BY_AMOUNT).add_list(keys, ratios, self._evaluation.get_name())


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--guess_tables', nargs='+', required=True, help="tables with the guesses of the user")
    parser.add_argument('--jsons', nargs='+', required=True, help='json file that contains the configuration')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    plot = ListPlot(RATIO_BY_AMOUNT)
    plot.set_y_axes([-1, 101])
    Plot.add_created_plot(RATIO_BY_AMOUNT, plot)
    plot2 = ListPlot(RATIO_BY_SIZES)
    plot2.set_y_axes([-1, 101])
    Plot.add_created_plot(RATIO_BY_SIZES, plot2)
    bar_plot = BarPlot(TOTAL_BY_SIZES)
    bar_plot.set_bar_width(0.005)
    bar_plot.set_y_axes([-1, 60])
    Plot.add_created_plot(TOTAL_BY_SIZES, bar_plot)
    for i in numpy.arange(0, len(args.guess_tables)):
        print(args.guess_tables[i])
        eva = Evaluation(args.guess_tables[i], args.jsons[i])
        ves = VesicleEvaluation(eva)
        ves.evaluate()
        ves.print_statistic()
    plot.draw()
    plot2.draw()
    bar_plot.draw()
    plt.show()
