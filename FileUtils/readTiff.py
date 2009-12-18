#!/usr/bin/python

##################
# readTiff.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Image
import numpy as np
from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon



def read3DTiff(filename):
    im = Image.open(filename)

    im.seek(0)

    #PIL's endedness support is subtly broken - try to fix it
    #NB this is untested for floating point tiffs
    endedness = 'LE'
    if im.ifd.prefix =='MM':
        endedness = 'BE'
    

    #ima = np.array(im.getdata(), 'int16').newbyteorder('BE')
    ima = np.array(im).newbyteorder(endedness)

    print ima.dtype
    
    #print ima.shape

    ima = ima.reshape((im.size[1], im.size[0], 1))

    pos = im.tell()

    try:
        while True:
            pos += 1
            im.seek(pos)

            #ima = np.concatenate((ima, np.array(im.getdata(), 'int16').newbyteorder('BE').reshape((im.size[1], im.size[0], 1))), 2)
            ima = np.concatenate((ima, np.array(im).newbyteorder(endedness).reshape((im.size[1], im.size[0], 1))), 2)

    except EOFError:
        pass

    return ima
