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
from collections import OrderedDict
from distutils.util import strtobool

import numpy
from numpy import distutils


class Evaluation(object):
    def __init__(self, guess_table, json_file_name):
        """
        
        Parameters
        ----------
        guess_table     contains only the answers of the user (no headlines etc.)
        json_file_name  contains the configuration json
        """
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

        for size in sums:
            absolute_good = sums[size]['good']
            absolute_wrong = sums[size]['wrong']
            absolute_total = absolute_good + absolute_wrong

            print('ratio size {}: {}% ({}/{})'.format(
                size,
                absolute_good * 100.0 / absolute_total, absolute_good, absolute_total))

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

        for size in sorted(sums):
            absolute_good = 0
            absolute_wrong = 0
            if 'good' in sums[size]:
                absolute_good = sums[size]['good']
            if 'wrong' in sums[size]:
                absolute_wrong = sums[size]['wrong']
            absolute_total = absolute_good + absolute_wrong

            print('ratio amount of points {}: {}% ({}/{})'.format(
                size,
                absolute_good * 100.0 / absolute_total, absolute_good, absolute_total))


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('guess_table', help="table with the guesses of the user")
    parser.add_argument('json', help='json file that contains the configuration')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    eva = Evaluation(args.guess_table, args.json)
    ves = VesicleEvaluation(eva)
    ves.evaluate()
    ves.print_statistic()
