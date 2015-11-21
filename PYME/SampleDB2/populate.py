#!/usr/bin/python

###############
# populate.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
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
################
#!/usr/bin/python
import sys
import os

#put the folder above this one on the python path
#print __file__
#print os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(os.path.split(os.path.abspath(__file__))[0])

#let Django know where to find the settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'SampleDB2.settings'

from samples.models import File


def addFiles(directory, extensions=['.h5r', '.h5']):
    #dir_size = 0
    for (path, dirs, files) in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] in extensions:
                filename = os.path.join(path, file)
                #print filename
                #try:
                f = File.GetOrCreate(filename)

                if f.filesize <=1: #was added before database updated
                    f.filesize = os.path.getsize(filename)
                    f.save()
                #except ValueError as e:
                #    print('Error is:')
                #    print(e)


if __name__ == '__main__':
    addFiles(sys.argv[1])
    
