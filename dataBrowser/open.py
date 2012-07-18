#!/usr/bin/python

###############
# open.py
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
from django.http import HttpResponse
import os
import sys
import subprocess


#def openFile(request, filename):
#    if not sys.platform == 'win32':
#        filename = '/' + filename
#    if os.path.exists(filename):
#        #return HttpResponse("Thumbnail for %s." % filename)
#        if filename.endswith('.h5') or filename.endswith('.kdf'):
#            if sys.platform == 'win32':
#                subprocess.Popen('..\\DSView\\dh5view.cmd %s' % (filename), shell=True)
#            else:
#                subprocess.Popen('../DSView/dh5view.py %s' % filename, shell=True)
#
#        elif filename.endswith('.h5r'):
#            if sys.platform == 'win32':
#                subprocess.Popen('..\\Analysis\\LMVis\\VisGUI.cmd %s' % (filename), shell=True)
#            else:
#                subprocess.Popen('../Analysis/LMVis/VisGUI.py %s' % filename, shell=True)
#
#
#    if not filename.startswith('/'):
#        filename = '/' + filename
#
#    #return HttpResponseRedirect('/browse' + '/'.join(filename.split('/')[:-1]))
#    return HttpResponseRedirect(request.META['HTTP_REFERER'])

def openFile(request, filename):
    if not filename.startswith('/'):
         filename = '/' + filename
         
    if filename.endswith('.pmu'):
        #strip url extension
        filename = filename[:-4]
    response = HttpResponse(mimetype="application/x-pyme-url")
    #response['Content-Disposition'] = 'attachment; filename=%s.pmu' % filename[1:]
    response.write('%s\n' % filename)
    response.write('''##This is a PYME url, and can be opened with the urlOpener application
    found in the PYME.FileUtils directory. Asociating this with the application/x-pyme-url
    MIME type should enable you to open files directly from the browser.

    urlOpener needs the environment variables PYMEMICRPATH and PYMENASPATH to find the
    the relavent files.''')
    return response

