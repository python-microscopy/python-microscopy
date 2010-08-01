##################
# driftio.py
#
# Copyright C Soeller 2010
# c.soeller@auckland.ac.nz
#
# This file may NOT be distributed without express permision from
# David Baddeley or Christian Soeller
#
##################

import cPickle

def saveDriftFile(filename,x,y,offs=0,step=1):
    fi = open(filename,'wb')
    cPickle.dump((x, y, offs, step),fi,2)
    fi.close()

def loadDriftFile(filename):
    fi = open(filename,'r')
    x, y, offs, step = cPickle.load(fi)
    fi.close()
    return x, y, offs, step

