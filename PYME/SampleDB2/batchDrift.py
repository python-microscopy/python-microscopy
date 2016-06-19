#!/usr/bin/python

###############
# batchDrift.py
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

from PYME.LMVis import pipeline
from PYMEnf.DriftCorrection import driftNoGUI

import sys
import os

sys.path.append(os.path.split(os.path.abspath(__file__))[0])

#let Django know where to find the settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'SampleDB2.settings'

from samples.models import DriftFit, Image

filename = sys.argv[1]

def fitDrift(filename):
    print(filename)
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
        
        print(df)
        
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
