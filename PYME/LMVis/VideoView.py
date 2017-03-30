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

from PYME.LMVis.View import View


class VideoView(View):
    JSON_DURATION = 'duration'

    def __init__(self, view_id, vec_up, vec_back, vec_right, translation, zoom, duration):
        """
        
        Parameters
        ----------
        view_id     is up to you, as long as serializable with json
        vec_up      np.array
        vec_back    np.array
        vec_right   np.array
        translation np.array
        zoom        usually a scalar
        duration    duration of the view transition in seconds
        """
        super(VideoView, self).__init__(view_id, vec_up, vec_back, vec_right, translation, zoom)
        self._duration = duration

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        if value:
            self._duration = value

    def to_json(self):
        ordered_dict = super(VideoView, self).to_json()
        ordered_dict[self.JSON_DURATION] = self._duration
        return ordered_dict

    @staticmethod
    def decode_json(json_obj):
        # if '__type__' in json_obj and json_obj['__type__'] == View:
        return VideoView(View.get_json_field(json_obj, View.JSON_VIEW_ID, 'id'),
                         View.get_json_array(json_obj, View.JSON_VEC_UP, numpy.array([0, 1, 0])),
                         View.get_json_array(json_obj, View.JSON_VEC_BACK, numpy.array([0, 0, 1])),
                         View.get_json_array(json_obj, View.JSON_VEC_RIGHT, numpy.array([1, 0, 0])),
                         View.get_json_array(json_obj, View.JSON_TRANSLATION, numpy.array([0, 0, 0])),
                         View.get_json_field(json_obj, View.JSON_ZOOM, 1),
                         View.get_json_field(json_obj, VideoView.JSON_DURATION, 1))


if __name__ == '__main__':
    view = View(1, numpy.array([1, 1, 1]), numpy.array([2, 2, 2]), numpy.array([3, 3, 3]),
                numpy.array([0, 0, 0]), 5)
    a = json.loads(json.dumps(view.to_json()))
    view2 = View.decode_json(a)
