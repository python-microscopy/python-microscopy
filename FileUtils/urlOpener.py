#!/usr/bin/python
import os
import sys
import subprocess
import re

from PYME.FileUtils import nameUtils

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
print PYMEDir
#print __file__


def openFile(url):
    if url.startswith(micrPrefix):
        filename = os.path.join(micrPath, *seps.split(url[len(micrPrefix):]))
    elif url.startswith(nasPrefix):
        filename = os.path.join(nasPath, *seps.split(url[len(nasPrefix):]))
    else:
        filename = url
        

    print filename

    if os.path.exists(filename):
        #return HttpResponse("Thumbnail for %s." % filename)
        #print 'pe'
        if filename.endswith('.h5') or filename.endswith('.kdf'):
            if sys.platform == 'win32':
                subprocess.Popen(PYMEDir + '\\DSView\\dh5view.cmd %s' % (filename), shell=True)
            else:
                subprocess.Popen(PYMEDir + '/DSView/dh5view.py %s' % filename, shell=True)

        elif filename.endswith('.h5r'):
            if sys.platform == 'win32':
                subprocess.Popen(PYMEDir + '\\Analysis\\LMVis\\VisGUI.cmd %s' % (filename), shell=True)
            else:
                #print 'foo'
                subprocess.Popen(PYMEDir + '/Analysis/LMVis/VisGUI.py %s' % filename, shell=True)
    
    
    
        

if __name__ == '__main__':
    f = open(sys.argv[1])
    url = f.readline().strip()
    f.close()
    #print url
    
    openFile(url)
