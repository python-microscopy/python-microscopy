#!/usr/bin/python

###############
# urlOpener.py
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
import os
import sys
import subprocess
import re

from PYME.IO.FileUtils import nameUtils

seps = re.compile('[\\\\/*]')

micrPrefix = '/mnt/MicrData/'
nasPrefix = '/mnt/NasData/'

micrPath = micrPrefix
if 'PYMEMICRPATH' in os.environ.keys():
    micrPath = os.environ['PYMEMICRPATH']

nasPath = nasPrefix
if 'PYMENASPATH' in os.environ.keys():
    nasPath = os.environ['PYMENASPATH']

PYMEDir = os.path.split(os.path.split(os.path.abspath(nameUtils.__file__))[0])[0]
print(PYMEDir)
#print __file__


def openFile(url):
    print(url)
        
    
    if url.startswith(micrPrefix):
        filename = os.path.join(micrPath, *seps.split(url[len(micrPrefix):]))
    elif url.startswith(nasPrefix):
        filename = os.path.join(nasPath, *seps.split(url[len(nasPrefix):]))
    else:
        filename = url
        

    print(filename)

    if os.path.exists(filename):
        #return HttpResponse("Thumbnail for %s." % filename)
        #print 'pe'
        if filename.endswith('.h5') or filename.endswith('.kdf'):
            if sys.platform == 'win32':
                subprocess.Popen('dh5view.cmd %s' % (filename), shell=True)
            else:
                subprocess.Popen('dh5view.py %s' % filename, shell=True)

        elif filename.endswith('.h5r'):
            if sys.platform == 'win32':
                subprocess.Popen('VisGUI.cmd %s' % (filename), shell=True)
            else:
                #print 'foo'
                subprocess.Popen('VisGUI.py %s' % filename, shell=True)
    
    
    
        

if __name__ == '__main__':
    #print url
    fn = ' '.join(sys.argv[1:])
    if fn.startswith('pyme:/'): #in gnome, whole url gets passed
        url = fn[6:]
    elif fn.startswith('//'): #in windows, just the bit after the : gets passed
        url = fn[1:]
    else:    
        f = open(fn)
        url = f.readline().strip()
        f.close()
    #print url
    
    openFile(url)
