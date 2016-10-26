#!/usr/bin/python

###############
# calcEventStats.py
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
import models
from PYME.LMVis.h5rNoGui import Pipeline
from PYME.Analysis.BleachProfile.kinModels import getPhotonNums
import numpy as np
import traceback

def getStatsChan(pipeline, chanName, file):
    p = pipeline
    p.colourFilter.setColour(chanName)
    if chanName == 'Everything':
        label = 'Everything'
    else:
        label = p.fluorSpeciesDyes[chanName]
    if 'Camera.CycleTime' in p.mdh.getEntryNames():
        t = p.colourFilter['t'].astype('f')*p.mdh.getEntry('Camera.CycleTime')
    else:
        t = p.colourFilter['t'].astype('f')*p.mdh.getEntry('Camera.IntegrationTime')
    nEvents = t.size
    tMax = t.max()
    tMedian = np.median(t)
    meanPhotons = getPhotonNums(p.colourFilter, p.mdh).mean()

    sts = models.EventStats(fileID=file, label=label, nEvents=nEvents, tMax=tMax, tMedian=tMedian, meanPhotons=meanPhotons)
    sts.save()
    return sts

def getStats(file):
    if file.filename.endswith('.h5r'):
        print((file.filename))
        try:
            p = Pipeline(file.filename)
            getStatsChan(p, 'Everything', file)
            
            chans = p.colourFilter.getColourChans()
            for c in chans:
                getStatsChan(p, c, file)

        except Exception as e:
            #traceback.print_exc()
            print(e)
        finally:
            try:
                p.selectedDataSource.resultsSource.close()
            except:
                pass

