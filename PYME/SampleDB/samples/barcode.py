#!/usr/bin/python

###############
# barcode.py
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
from PIL import Image
import numpy as np

def int2bin(n, count=24):
    """returns the binary of integer n, using count number of digits"""
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def barcode2d(request, idNum):
    response = HttpResponse(mimetype="image/png")
    idNum=int(idNum)
    #bid = '%64s' %(bin(idNum).split('b')[1])
    bid = int2bin(idNum, 64)
    #print idNum, bid
    im = 255*np.array([c=='1' for c in bid]).reshape(4,-1)
    #print im
    Image.fromarray(im.astype('uint8')).save(response, 'PNG')
    return response
    
