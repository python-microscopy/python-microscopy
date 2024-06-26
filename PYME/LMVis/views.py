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
from collections import OrderedDict

import numpy
import numpy as np

try:
    # Put this in a try-except clause as a) the quaternion module is not packaged yet and b) it has a dependency on a recent numpy version
    # so we might not want to make it a dependency yet.
    import quaternion
    HAVE_QUATERNION = False#True
except ImportError:
    print('quaternion module not found, disabling custom clip plane orientations')
    HAVE_QUATERNION = False


clipping_dtype = [('x', '<f4', (2,)), ('y', '<f4', (2,)), ('z', '<f4', (2,)), ('v', '<f4', (2,))]
dummy_clipping = np.array([-1e6, 1e6, -1e6, 1e6, -1e6, 1e6, -1e6, 1e6], 'f4').view(clipping_dtype)

def_clip_plane_orientation = np.array([1,0,0,0], 'f8')#np.eye(4,4,dtype='f')

class View(object):
    def __init__(self, view_id='id', vec_up=[0,1,0], vec_back = [0,0,1], vec_right = [1,0,0], translation= [0,0,0],
                 scale=1, clipping = dummy_clipping, clip_plane_orientation=def_clip_plane_orientation, clip_plane_position=[0,0,0],  **kwargs):
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
        self.vec_up = np.array(vec_up)
        self.vec_back = np.array(vec_back)
        self.vec_right = np.array(vec_right)
        self.translation = np.array(translation)
        self.scale = scale
        if isinstance(clipping, np.ndarray) and (clipping.dtype == clipping_dtype):
            self.clipping = np.copy(clipping)
        else:
            self.clipping= np.array(clipping, 'f4').squeeze().view(clipping_dtype)
            
        if HAVE_QUATERNION:
            if isinstance(clip_plane_orientation, quaternion.quaternion):
                self.clip_plane_orientation = np.copy(clip_plane_orientation)
            else:
                self.clip_plane_orientation = quaternion.quaternion(*clip_plane_orientation)
        else:
            #this won't work properly, but will stop us firing an error
            self.clip_plane_orientation = np.array(clip_plane_orientation, dtype='f8')
            
        self.clip_plane_position = np.array(clip_plane_position)

    @property
    def view_id(self):
        return self._view_id

    @view_id.setter
    def view_id(self, value):
        if value:
            self._view_id = value

    # @property
    # def vec_up(self):
    #     return self._vec_up
    #
    # @property
    # def vec_back(self):
    #     return self._vec_back
    #
    # @property
    # def vec_right(self):
    #     return self._vec_right
    #
    # @property
    # def translation(self):
    #     return self._translation
    #
    # @property
    # def zoom(self):
    #     return self._zoom

    def __add__(self, other):
        return View(None,
                    self.vec_up + other.vec_up,
                    self.vec_back + other.vec_back,
                    self.vec_right + other.vec_right,
                    self.translation + other.translation,
                    self.scale + other.scale,
                    (self.clipping.view('8f4') + other.clipping.view('8f4')).squeeze(),
                    self.clip_plane_orientation + other.clip_plane_orientation,
                    self.clip_plane_position + other.clip_plane_position
                    )

    def __sub__(self, other):
        return View(None,
                    self.vec_up - other.vec_up,
                    self.vec_back - other.vec_back,
                    self.vec_right - other.vec_right,
                    self.translation - other.translation,
                    self.scale - other.scale,
                    (self.clipping.view('8f4') - other.clipping.view('8f4')).squeeze(),
                    self.clip_plane_orientation -self.clip_plane_orientation,
                    self.clip_plane_position - other.clip_plane_position
                    )

    def __mul__(self, scalar):
        return View(None,
                    self.vec_up * scalar,
                    self.vec_back * scalar,
                    self.vec_right * scalar,
                    self.translation * scalar,
                    self.scale * scalar,
                    (self.clipping.view('8f4') * scalar).squeeze(),
                    self.clip_plane_orientation * scalar,
                    self.clip_plane_position*scalar
                    )

    def __div__(self, scalar):
        return View(None,
                    self.vec_up / scalar,
                    self.vec_back / scalar,
                    self.vec_right / scalar,
                    self.translation / scalar,
                    self.scale / scalar,
                    (self.clipping.view('8f4') /scalar).squeeze(),
                    self.clip_plane_orientation / scalar,
                    self.clip_plane_position / scalar
                    )

    __truediv__ = __div__ #div goes away on py3

    def normalize_view(self):
        self.vec_up = self.normalize(self.vec_up)
        self.vec_back = self.normalize(self.vec_back)
        self.vec_right = self.normalize(self.vec_right)
        return self

    @staticmethod
    def normalize(array):
        return array / numpy.linalg.norm(array)

    def to_json(self):
        ordered_dict = dict()# OrderedDict()
        ordered_dict['view_id'] = self.view_id
        ordered_dict['vec_up'] = self.vec_up.tolist()
        ordered_dict['vec_back'] = self.vec_back.tolist()
        ordered_dict['vec_right'] = self.vec_right.tolist()
        ordered_dict['translation'] = self.translation.tolist()
        ordered_dict['scale'] = float(self.scale)
        ordered_dict['clipping'] = self.clipping.view('8f4').squeeze().tolist()
        #ordered_dict['clip_plane_orientation'] = self.clip_plane_orientation.view('4f8').squeeze().tolist()
        ordered_dict['clip_plane_position'] = self.clip_plane_position.tolist()

        return ordered_dict

    def __str__(self):
        return str(self.to_json())

    @classmethod
    def decode_json(cls, json_obj):
        # if '__type__' in json_obj and json_obj['__type__'] == View:
        return cls(**json_obj)
    
    @classmethod
    def copy(cls, view):
        return cls.decode_json(view.to_json())
    
    def top(self):
        """ set the view to top-down view"""
        self.vec_up = np.array([0,1,0])
        self.vec_back = np.array([0,0,1])
        self.vec_right = np.array([1,0,0])
        
    def front(self):
        """ set the view to front view"""
        self.vec_up = np.array([0,0, -1])
        self.vec_back = np.array([0,-1,0])
        self.vec_right = np.array([1,0,0])

    def right(self):
        """ set the view to right view"""
        self.vec_up = np.array([1, 0, 0])
        self.vec_back = np.array([0, -1,0])
        self.vec_right = np.array([0,0,1])    



    
#TOP = View()

class VideoView(View):
    JSON_DURATION = 'duration'
    
    def __init__(self, view_id='id', vec_up=[0,1,0], vec_back = [0,0,1], vec_right = [1,0,0], translation= [0,0,0], scale=1, clipping=dummy_clipping, duration = 1, **kwargs):
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
        super(VideoView, self).__init__(view_id, vec_up, vec_back, vec_right, translation, scale, clipping, **kwargs)
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
    
    # @staticmethod
    # def decode_json(json_obj):
    #     # if '__type__' in json_obj and json_obj['__type__'] == View:
    #     return VideoView(View.get_json_field(json_obj, View.JSON_VIEW_ID, 'id'),
    #                      View.get_json_array(json_obj, View.JSON_VEC_UP, numpy.array([0, 1, 0])),
    #                      View.get_json_array(json_obj, View.JSON_VEC_BACK, numpy.array([0, 0, 1])),
    #                      View.get_json_array(json_obj, View.JSON_VEC_RIGHT, numpy.array([1, 0, 0])),
    #                      View.get_json_array(json_obj, View.JSON_TRANSLATION, numpy.array([0, 0, 0])),
    #                      View.get_json_field(json_obj, View.JSON_ZOOM, 1),
    #                      View.get_json_field(json_obj, VideoView.JSON_DURATION, 1))


if __name__ == '__main__':
    view = View(1, numpy.array([1, 1, 1]), numpy.array([2, 2, 2]), numpy.array([3, 3, 3]),
                numpy.array([0, 0, 0]), 5)
    a = json.loads(json.dumps(view.to_json()))
    view2 = View.decode_json(a)
