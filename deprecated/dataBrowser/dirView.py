#!/usr/bin/python

###############
# dirView.py
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
import os.path
from django.http import HttpResponse
from django.shortcuts import render_to_response
import os
import sys
import glob
from PYME.misc.dirSize import getDirectorySize
import numpy

MINSIZE=1
ONEGSIZE=2

def sortDirs(dirs):
    mangledDirnames =[]
    for d in dirs:
        dparts = d['name'].split('-')
        mangledDirnames.append('-'.join(s.zfill(2) for s in dparts[::-1]))

    dirI = numpy.argsort(mangledDirnames)

    return list(numpy.array(dirs)[dirI])

def findAnalysis(filename):
    if not filename.endswith('.h5'):
        return None
    else:
        fileparts = filename.split('/')

        cand = '/'.join(fileparts[:-2] + ['analysis',] + fileparts[-2:]) + 'r'
        #print cand

        if os.path.exists(cand):
            if not cand.startswith('/'):
                cand = '/' + cand
            return cand
        else:
            return None

def viewdir(request, dirname):
    if dirname.endswith('/'):
        dirname = dirname[:-1]
    if not sys.platform == 'win32':
        dirname1 = '/' + dirname
    else:
        dirname1 = dirname
    if os.path.exists(dirname1):
        #return HttpResponse("Thumbnail for %s." % filename)
        children = glob.glob(dirname1 + '/*')

        children = ['/'.join(c.split('\\')) for c in children]

        files = [{'name':os.path.split(f)[1], 'size': '%.2f MB' % (os.path.getsize(f)/1024.**2), 'analysis':findAnalysis(f)} for f in children if os.path.isfile(f)]
        dirs = [{'name':os.path.split(f)[1], 'size': '%.2f GB' % (getDirectorySize(f)/1024.**3), 'sizef': MINSIZE + numpy.sqrt(ONEGSIZE*getDirectorySize(f)/1024.**3)} for f in children if os.path.isdir(f)]

        root_dirs = dirname.split('/')

        dirs = sortDirs(dirs)

        rdstub = ''
        rdirs = []
        for rd in root_dirs:
            if len(rd) > 0:
                rdstub = rdstub + '/' + rd
                rdirs.append({'name':rd, 'path':rdstub})


        return render_to_response('templates/dirview.html', {'dirname':dirname, 'files': files, 'dirs':dirs, 'rootdirs':rdirs})
    else:
        return HttpResponse("Directory %s does not exist." % dirname)
