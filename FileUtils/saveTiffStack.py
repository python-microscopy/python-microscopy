#!/usr/bin/python

##################
# saveTiffStack.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Image
from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon
import subprocess

def writeTiff(im, outfile):
    command = ["tiffcp"]
    # add options here, if any (e.g. for compression)

    #im = im.astype('uint16')
    im = im.astype('>u2').astype('<u2')

    for i in range(im.shape[2]):
        framefile = "/tmp/frame%d.tif" % i

        Image.fromarray(im[:,:,i].squeeze(), 'I;16').save(framefile)
        command.append(framefile)

    command.append(outfile)
    subprocess.call(command)

    # remove frame files here
    subprocess.call('rm /tmp/frame*.tif', shell=True)


#Adapted to use 3rd dimension as z from priithons 'useful.py'
def saveTiffMultipage(arr, fn, **params):
    if arr.ndim == 2:
        #fake a 3rd dimension
        #raise ValueError, "can only save 3d arrays"
        arr = arr[:,:,None]

    fp = open(fn, 'w+b')

    ifd_offsets=[]

    if arr.dtype == 'uint16':
        nptype = 'uint16'
        piltype = 'I;16'
    else:
        nptype = 'f'
        piltype = 'F'

    params["_debug_multipage"] = True
    for z in range(arr.shape[2]):
        ii = Image.fromarray(arr[:,:,z].astype(nptype), piltype)

        fp.seek(0,2) # go to end of file
        if z==0:
            # ref. PIL  TiffImagePlugin
            # PIL always starts the first IFD at offset 8
            ifdOffset = 8
        else:
            ifdOffset = fp.tell()

        ii.save(fp, format="TIFF", **params)

        if z>0: # correct "next" entry of previous ifd -- connect !
            ifdo = ifd_offsets[-1]
            fp.seek(ifdo)
            ifdLength = ii._debug_multipage.i16(fp.read(2))
            fp.seek(ifdLength*12,1) # go to "next" field near end of ifd
            fp.write(ii._debug_multipage.o32( ifdOffset ))

        ifd_offsets.append(ifdOffset)
    fp.close()
