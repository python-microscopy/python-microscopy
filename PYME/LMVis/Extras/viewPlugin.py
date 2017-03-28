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
from wx import wx
import json

from PYME.LMVis.View import View


class ViewHandler(object):

    def __init__(self, vis_fr):
        vis_fr.AddMenuItem('Extras>View', 'Save View', lambda e: self.save_view(vis_fr.glCanvas))
        vis_fr.AddMenuItem('Extras>View', 'Load View', lambda e: self.load_view(vis_fr.glCanvas))

    @staticmethod
    def save_view(canvas):
        view = canvas.get_view()
        file_name = wx.FileSelector('Save view as json named... ')
        if file_name:
            if not file_name.endswith('.json'):
                file_name = '{}.json'.format(file_name)
            with open(file_name, 'w') as f:
                f.writelines(view.to_json())

    @staticmethod
    def load_view(canvas):
        file_name = wx.FileSelector('Open View-JSON file')
        if file_name:
            with open(file_name, 'r') as f:
                view = View.decode_json(json.load(f))
            canvas.set_view(view)


def Plug(vis_fr):
    ViewHandler(vis_fr)
