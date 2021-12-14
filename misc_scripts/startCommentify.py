#!/usr/bin/python

##################
# startCommentify.py
#
# Copyright David Baddeley, 2009
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
##################

import os

copyr = """#!/usr/bin/python

##################
# %s
#
# Copyright David Baddeley, 2009
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
##################

"""

def walkFunc(arg, dirname, fnames):
    for fn in fnames:
        if fn.endswith('.py'):
          f = open(os.path.join(dirname, fn), 'r')

          text = f.read()

          f.close()

          #f = open()

          print('------------------------------------------------')
          print(fn)
          print('------------------------------------------------\n')

          print((text[:500]))

          res = raw_input('Add header to file? y/n:')

          if res.upper() == 'Y':
              f = open(os.path.join(dirname, fn), 'w')
              f.write(((copyr % fn) + text))
              f.close()




if __name__ == '__main__':  
    os.path.walk('/home/david/PYME/PYME', walkFunc, 0)

