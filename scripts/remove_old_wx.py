#!/usr/bin/python

###############
# remove_old_wx.py
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


import os
import glob
import shutil
from distutils.sysconfig import get_python_lib


def main():
    wxFiles = glob.glob(os.path.join(get_python_lib(), 'wx*'))
    
    for f in wxFiles:
        if os.path.isdir(f):
            #recursively delete directory
            shutil.rmtree(f)
        else:
            os.remove(f)
            
if __name__ == '__main__':
    main()