#!/usr/bin/python

# videoPlugin.py
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
import cv2
import numpy
from cv2.cv import CV_FOURCC
from wx import wx


class VideoPlugin:

    JSON_LIST_NAME = 'views'

    def __init__(self, vis_fr):
        self.views = []
        self.width = 0
        self.height = 0
        vis_fr.AddMenuItem('Extras>Video', 'Save View', lambda e: self.add_view(vis_fr.glCanvas))
        vis_fr.AddMenuItem('Extras>Video', 'Save View List', lambda e: self.save_view_list())
        vis_fr.AddMenuItem('Extras>Video', 'Make', lambda e: self.make_video(vis_fr.glCanvas))

    def add_view(self, canvas):
        self.views.append(canvas.get_view())
        self.width = canvas.Size[0]
        self.height = canvas.Size[1]
        pass

    def save_view_list(self):
        file_name = wx.FileSelector('Save view as json named... ')
        if file_name:
            if not file_name.endswith('.json'):
                file_name = '{}.json'.format(file_name)
            with open(file_name, 'w') as f:
                f.write('{')
                f.write('{}:['.format(self.JSON_LIST_NAME))
                is_first = True
                for view in self.views:
                    if not is_first:
                        f.write(',')
                    f.writelines(view.to_json())
                    is_first = False
                f.write(']}')

    def load_view_list(self, canvas):
        pass

    def make_video(self, canvas):
        # file_name = wx.FileSelector('Save video as avi named... ')
        # TODO remove default filename
        file_name = u'C:\\Users\\mmg82\\Desktop\\test.avi'

        video = cv2.VideoWriter(file_name, -1, 30, (self.width, self.height))
        print('{},{}'.format(self.height, self.width))
        if not self.views:
            self.add_view(canvas)
        current_view = None
        for view in self.views:
            if not current_view:
                current_view = view
            else:
                steps = 40
                difference_view = (view - current_view)/steps
                for step in range(0, steps):
                    new_view = current_view+difference_view*step
                    canvas.set_view(new_view)
                    img = numpy.fromstring(canvas.getIm().tostring(), numpy.ubyte).reshape(self.height, self.width, 3)
                    video.write(cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR))
                current_view = view
        video.release()
        print("finished")

    @staticmethod
    def PIL2array(img):
        return numpy.array(img.getdata(),
                           numpy.ubyte).reshape(img.size[1], img.size[0], 3)


def Plug(vis_fr):
    VideoPlugin(vis_fr)


