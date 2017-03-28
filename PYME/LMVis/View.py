#!/usr/bin/python

# View.py
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
import json


class View(object):

    def __init__(self, view_id, vec_up, vec_back, vec_right, translation, zoom):
            self._view_id = view_id
            self._vec_up = vec_up
            self._vec_back = vec_back
            self._vec_right = vec_right
            self._translation = translation
            self._zoom = zoom

    def __add__(self, other):
        return View(None,
                    [a_i + b_i for a_i, b_i in zip(self._vec_up, other._vec_up)],
                    [a_i + b_i for a_i, b_i in zip(self._vec_back, other._vec_back)],
                    [a_i + b_i for a_i, b_i in zip(self._vec_right, other._vec_right)],
                    [a_i + b_i for a_i, b_i in zip(self._translation, other._translation)],
                    self._zoom + other._zoom
                    )

    def __sub__(self, other):
        return View(None,
                    [a_i - b_i for a_i, b_i in zip(self._vec_up, other._vec_up)],
                    [a_i - b_i for a_i, b_i in zip(self._vec_back, other._vec_back)],
                    [a_i - b_i for a_i, b_i in zip(self._vec_right, other._vec_right)],
                    [a_i - b_i for a_i, b_i in zip(self._translation, other._translation)],
                    self._zoom - other._zoom
                    )

    def __mul__(self, scalar):
        return View(None,
                    [a_i * scalar for a_i in self._vec_up],
                    [a_i * scalar for a_i in self._vec_back],
                    [a_i * scalar for a_i in self._vec_right],
                    [a_i * scalar for a_i in self._translation],
                    self._zoom*scalar
                    )

    def __div__(self, scalar):
        return View(None,
                    [a_i / scalar for a_i in self._vec_up],
                    [a_i / scalar for a_i in self._vec_back],
                    [a_i / scalar for a_i in self._vec_right],
                    [a_i / scalar for a_i in self._translation],
                    self._zoom / scalar
                    )

    def to_json(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def decode_json(json_obj):
        # if '__type__' in json_obj and json_obj['__type__'] == View:
            return View(json_obj['_view_id'],
                        json_obj['_vec_up'],
                        json_obj['_vec_back'],
                        json_obj['_vec_right'],
                        json_obj['_translation'],
                        json_obj['_zoom'])
