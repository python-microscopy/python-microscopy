#!/usr/bin/python

##################
# LMimg.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

# LMimg: a class to process, render and save LM images

# example usage:

# from PYME.Analysis import LMimg
# from pylab import *
# resname = '/Users/csoelle/analysis/data/14_8_series_A.h5r'
# import tables
# h5r = tables.openFile(resname)
# lmtest = LMimg.LMimg(h5r)
# rg = lmtest.rGauss([10,10])
# rqt = lmtest.rQT([10,10])
# rg.save('rgauss.tif')
# imshow(rqt.img,cmap=cm.hot)

import LMVis.rendGauss as LMrg
import QuadTree.pointQT as LMpqt
from QuadTree.rendQT import rendQTan
import MetaData

import Image
import numpy
import math

class ViewPort:
    def __init__(self,xmin,xmax,ymin,ymax,zmin=0,zmax=0,unit='um'):
	self.xmin = xmin
	self.xmax = xmax
	self.ymin = ymin
	self.ymax = ymax
	self.zmin = zmin
	self.zmax = zmax
	self.unit = unit

class LMrend:
    def __init__(self, img, mode, md, vp, nevents=-1):
	self.img = img
	self.mode  = mode
	self.md = md
	self.vp = vp
	self.nevents = nevents

    def save(self,name):
	img = Image.fromarray(self.img,mode="F")
	img.save(name,format="TIFF")
	
def ceilLim(x,unit=1e3):
    return unit*math.ceil(x/unit)

def floorLim(x,unit=1e3):
    return unit*math.floor(x/unit)

class LMimg:
    def __init__(self,fitfile,md=MetaData.TIRFDefault):  # metadata should come from file in future!
	self.fitfile = fitfile
	self.md = md
	try:
	    self.fitresults = self.fitfile.root.FitResults.read()
	except:
	    print "Could not obtain fitresults from File, is this a valid HDF fitresults file?"
	    self.fitresults = None

    def rGauss(self,res=[20,20]):
	x0 = self.fitresults[:]['fitResults']['x0']
	y0 = self.fitresults[:]['fitResults']['y0']
	sx = self.fitresults[:]['fitError']['x0']
	vp = ViewPort(floorLim(x0.min()),ceilLim(x0.max()),floorLim(y0.min()),ceilLim(y0.max()),unit='nm')
	img = LMrg.rendGaussP(x0,y0,sx,
			      numpy.arange(vp.xmin,vp.xmax,res[0]),
			      numpy.arange(vp.ymin,vp.ymax,res[1]))
	md = MetaData.MetaData(MetaData.VoxelSize(res[0],res[1],700,unit='nm'))
	nevts = len(x0)
	rd = LMrend(img,"Gauss",md,vp,nevents=nevts)
	# normalize image so that img.sum() == nevents
	rd.img /= rd.img.sum()
	rd.img *= nevts
	return rd

    def rQT(self,res=[20,20]):
	x0 = self.fitresults[:]['fitResults']['x0']
	y0 = self.fitresults[:]['fitResults']['y0']
	vp = ViewPort(floorLim(x0.min()),ceilLim(x0.max()),floorLim(y0.min()),ceilLim(y0.max()),unit='nm')
	qt = LMpqt.qtRoot(vp.xmin,vp.xmax,vp.ymin,vp.ymax)
	for i in range(len(x0)):
	    qt.insert(LMpqt.qtRec(x0[i],y0[i]," "))
	nx = round((vp.xmax-vp.xmin)/res[0])
	ny = round((vp.ymax-vp.ymin)/res[1])
	imqt = numpy.zeros((nx,ny),'f')
	rendQTan(imqt,qt) # render into provided image container
	md = MetaData.MetaData(MetaData.VoxelSize(res[0],res[1],700,unit='nm'))
	rd = LMrend(imqt,"QT",md,vp,nevents=len(x0))
	rd.qt = qt
	return rd

