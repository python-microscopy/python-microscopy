#!/usr/bin/python

# viewPlugin.py
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
from wx import wx
import json


class View(object):

    def __init__(self, view_id, vec_up, vec_back, vec_right, translation, zoom):
            self._view_id = view_id
            self._vec_up = vec_up
            self._vec_back = vec_back
            self._vec_right = vec_right
            self._translation = translation
            self._zoom = zoom

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


class ViewHandler(object):

    def __init__(self, vis_fr):
        vis_fr.AddMenuItem('Extras>View', 'Save View', lambda e: self.save_view(vis_fr.glCanvas))
        vis_fr.AddMenuItem('Extras>View', 'Load View', lambda e: self.load_view(vis_fr.glCanvas))

    def save_view(self, canvas):
        view = View('id',
                    canvas.vecUp.tolist(),
                    canvas.vecBack.tolist(),
                    canvas.vecRight.tolist(),
                    [canvas.xc, canvas.yc, canvas.zc],
                    canvas.scale)
        file_name = wx.FileSelector('Save view as json named... ')
        if file_name:
            with open('{}.json'.format(file_name), 'w') as f:
                f.writelines(view.to_json())

    def load_view(self, canvas):
        file_name = wx.FileSelector('Open View-JSON file')
        if file_name:
            with open(file_name, 'r') as f:
                view = View.decode_json(json.load(f))

            canvas.vecBack = numpy.array(view._vec_back)
            canvas.vecRight = numpy.array(view._vec_right)
            canvas.vecUp = numpy.array(view._vec_up)
            canvas.xc = view._translation[0]
            canvas.yc = view._translation[1]
            canvas.zc = view._translation[2]
            canvas.scale = view._zoom
            canvas.Refresh()


def Plug(vis_fr):
    ViewHandler(vis_fr)
