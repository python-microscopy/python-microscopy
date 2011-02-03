#!/usr/bin/python

##################
# ft.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Pyro.core
import sys
sys.path.append('/home/david/pysmi_simulator/py_fit')
import remFitHDF
import os
import MetaData

tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')

seriesName = filenames[0].split(os.sep)[-1][:-9]

def pushImages(startingAt=0):
    for i in range(len(filenames)):
        tq.postTask(remFitFromFilename.fitTask(filenames[i], seriesName, .9, MetaData.TIRFDefault, 'LatGaussFitF', bgfiles=filenames[max(i-10, 0):i], SNThreshold=True))

import fitIO



