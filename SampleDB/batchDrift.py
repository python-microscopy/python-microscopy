# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:39:29 2012

@author: david
"""
from PYME.Analysis.LMVis import pipeline
from PYMEnf.DriftCorrection import driftNoGUI

import sys
import os

sys.path.append(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])

#let Django know where to find the settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'SampleDB.settings'

from SampleDB.samples.models import DriftFit, Image

filename = sys.argv[1]

def fitDrift(filename):
    print filename
    pipe = pipeline.Pipeline(filename)

    im = Image.objects.get(pk=pipe.mdh['imageID'])
    if im.drift_settings.count() < 1:    
        dc = driftNoGUI.DriftCorrector(pipe.filter)
        dc.SetNumPiecewiseSegments(5)
        
        dc.FitDrift()
        
        df = DriftFit(imageID=im, exprX=dc.driftExprX, 
                      exprY=dc.driftExprY, exprZ=dc.driftExprZ,
                      parameters=dc.driftCorrParams, auto=True)
        df.save()
        
        print df
        
    pipe.CloseFiles()


def procFiles(directory, extensions=['.h5r']):
    #dir_size = 0
    for (path, dirs, files) in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] in extensions:
                filename = os.path.join(path, file)
                #print filename
                fitDrift(filename)


if __name__ == '__main__':
    procFiles(sys.argv[1])