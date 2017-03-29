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

import numpy


class View(object):

    def __init__(self, view_id, vec_up, vec_back, vec_right, translation, zoom):
        """
        
        Parameters
        ----------
        view_id     is up to you, as long as serializable with json
        vec_up      np.array
        vec_back    np.array
        vec_right   np.array
        translation np.array
        zoom        usually a scalar
        """
        super(View, self).__init__()
        self._view_id = view_id
        self._vec_up = vec_up
        self._vec_back = vec_back
        self._vec_right = vec_right
        self._translation = translation
        self._zoom = zoom
    
    @property
    def view_id(self):
        return self._view_id

    @property
    def vec_up(self):
        return self._vec_up

    @property
    def vec_back(self):
        return self._vec_back

    @property
    def vec_right(self):
        return self._vec_right

    @property
    def translation(self):
        return self._translation

    @property
    def zoom(self):
        return self._zoom

    def __add__(self, other):
        return View(None,
                    [a_i + b_i for a_i, b_i in zip(self.vec_up, other.vec_up)],
                    [a_i + b_i for a_i, b_i in zip(self.vec_back, other.vec_back)],
                    [a_i + b_i for a_i, b_i in zip(self.vec_right, other.vec_right)],
                    [a_i + b_i for a_i, b_i in zip(self.translation, other.translation)],
                    self._zoom + other.zoom
                    )

    def __sub__(self, other):
        return View(None,
                    [a_i - b_i for a_i, b_i in zip(self.vec_up, other.vec_up)],
                    [a_i - b_i for a_i, b_i in zip(self.vec_back, other.vec_back)],
                    [a_i - b_i for a_i, b_i in zip(self.vec_right, other.vec_right)],
                    [a_i - b_i for a_i, b_i in zip(self.translation, other.translation)],
                    self._zoom - other.zoom
                    )

    def __mul__(self, scalar):
        return View(None,
                    [a_i * scalar for a_i in self.vec_up],
                    [a_i * scalar for a_i in self.vec_back],
                    [a_i * scalar for a_i in self.vec_right],
                    [a_i * scalar for a_i in self.translation],
                    self._zoom*scalar
                    )

    def __div__(self, scalar):
        return View(None,
                    [a_i / scalar for a_i in self.vec_up],
                    [a_i / scalar for a_i in self.vec_back],
                    [a_i / scalar for a_i in self.vec_right],
                    [a_i / scalar for a_i in self.translation],
                    self._zoom / scalar
                    )

    def to_json(self):
        return json.dumps(self, cls=ViewEncoder)

    @staticmethod
    def decode_json(json_obj):
        # if '__type__' in json_obj and json_obj['__type__'] == View:
        return View(json_obj[ViewEncoder.JSON_VIEW_ID],
                    numpy.array(json_obj[ViewEncoder.JSON_VEC_UP]),
                    numpy.array(json_obj[ViewEncoder.JSON_VEC_BACK]),
                    numpy.array(json_obj[ViewEncoder.JSON_VEC_RIGHT]),
                    numpy.array(json_obj[ViewEncoder.JSON_TRANSLATION]),
                    json_obj[ViewEncoder.JSON_ZOOM])


class ViewEncoder(json.JSONEncoder):

    JSON_VIEW_ID = 'view_id'
    JSON_VEC_UP = 'vec_up'
    JSON_VEC_BACK = 'vec_back'
    JSON_VEC_RIGHT = 'vec_right'
    JSON_TRANSLATION = 'translation'
    JSON_ZOOM = 'zoom'

    def default(self, obj):
        if isinstance(obj, View):
            return {self.JSON_VIEW_ID: obj.view_id,
                    self.JSON_VEC_UP: obj.vec_up.tolist(),
                    self.JSON_VEC_BACK: obj.vec_back.tolist(),
                    self.JSON_VEC_RIGHT: obj.vec_right.tolist(),
                    self.JSON_TRANSLATION: obj.translation.tolist(),
                    self.JSON_ZOOM: obj.zoom}
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    view = View(1, numpy.array([1, 1, 1]), numpy.array([2, 2, 2]), numpy.array([3, 3, 3]),
                numpy.array([0, 0, 0]), 5)
    a = json.loads(json.dumps(view, cls=ViewEncoder))
    view2 = View.decode_json(a)
