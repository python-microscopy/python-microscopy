#!/usr/bin/python

###############
# thumbnails.py
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
from PIL import Image
    
from PYME.FileUtils import thumbnailDatabase

def thumb(request, filename, size=200):
    if 'size' in request.REQUEST:
        size = request.REQUEST['size']
    if not sys.platform == 'win32':
        filename = '/' + filename
    if os.path.exists(filename):
        #return HttpResponse("Thumbnail for %s." % filename)
        
        im = thumbnailDatabase.getThumbnail(filename)
        if im == None:
            return HttpResponse("Could not generate thumbnail for %s." % filename)

        response = HttpResponse(mimetype="image/png")

        xsize = im.shape[0]
        ysize = im.shape[1]

        zoom = float(size)/max(xsize, ysize)

        Image.fromarray(im).resize((int(zoom*ysize), int(zoom*xsize))).save(response, 'PNG')
        return response
    else:
        return HttpResponse("File %s does not exist." % filename)
