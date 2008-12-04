#!/usr/bin/python
'''Script to find and measure small objects in a 3D confocal data set. Call as follows: pointFitConfoc.py inFile threshold outFile'''
from PYME.FileUtils import readTiff
from PYME.Analysis import MetaData
from PYME.Analysis.ofind import ObjectIdentifier
from PYME.Analysis.FitFactories.LatGaussFitF import FitFactory

import os
import sys
from numpy import sqrt

#set up parameters
md = MetaData.TIRFDefault

#Voxel sizes - alter as appropriate, in um
md.voxelsize.x = 0.09
md.voxelsize.y = 0.09
md.voxelsize.z = 0.20

#Hack the 'ccd' so we don't have to worry about the ADOffset preventing
#fits from working. FIXME: find sensible values for all parameter for confocal
#imaging so we can get error estimates.
md.CCD.ADOffset=0


#set this to the measured PSF FWHM. The correct way measure the PSF 
#is to put the nominal bead FWHM (bead diameter/sqrt(2)) in here and analyse 
#the beads. The median 'corrected' FWHM is then that of the microscope PSF.
scopeFWHM = 250 #in nm


#'default' threshold value for object identification- you probably want to 
#enter this on the command line instead
ofindThreshold = 20e3

####################################
#End of user-modifiable parameters #
####################################


if __name__ == '__main__':

    if len(sys.argv) >= 2: #we were passed a filename
        filename = sys.argv[1]
    else:# we'll have to create a GUI and ask
        import wx
        app = wx.App()
        filename = wx.FileSelector('Choose an image file', default_extension='.tif')

    if len(sys.argv) >=3: #we've got a threshold
        ofindThreshold = float(sys.argv[2])

    
    if len(sys.argv) >=4: #we've got a name for the output file
        outFilename = sys.argv[3]
    else: #build an output filename from input
        outFilename = filename.split('.')[0] + '.txt'

    if os.path.exists(outFilename): #don't overwrite an existing file
        raise "Output file '%s' already exists" % outFilename
        
    
    #read data
    ima = readTiff.read3DTiff(filename)

    #create an object identifier
    ofd = ObjectIdentifier(ima)
    #and identify objects ...
    ofd.FindObjects(ofindThreshold)


    #create a fit factory
    ff = FitFactory(ima, md)

    #iterate over found objects, fitting at each point
    res = [ff.FromPoint(p.x, p.y) for p in ofd]

    of = open(outFilename, 'w')

    of.write('ID\tx\ty\tA\tsigma\tcorrectedFWHM\n')

    for r, i in zip(res, range(len(res))):
        of.write('%d\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.0f\n' % (i, r.x0()/1e3, r.y0()/1e3, r.A(), r.sigma(), sqrt((2.35*r.sigma())**2 - scopeFWHM**2)))

    of.close()
    
    
