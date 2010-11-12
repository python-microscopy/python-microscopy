import models
from PYME.Analysis.LMVis.h5rNoGui import Pipeline
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
        t = p.colourFilter['t']*p.mdh.getEntry('Camera.CycleTime')
    else:
        t = p.colourFilter['t']*p.mdh.getEntry('Camera.IntegrationTime')
    nEvents = t.size
    tMax = t.max()
    tMedian = np.median(t)
    meanPhotons = getPhotonNums(p.colourFilter, p.mdh).mean()

    sts = models.EventStats(fileID=file, label=label, nEvents=nEvents, tMax=tMax, tMedian=tMedian, meanPhotons=meanPhotons)
    sts.save()
    return sts

def getStats(file):
    if file.filename.endswith('.h5r'):
        print file.filename
        try:
            p = Pipeline(file.filename)
            getStatsChan(p, 'Everything', file)
            
            chans = p.colourFilter.getColourChans()
            for c in chans:
                getStatsChan(p, c, file)

        except Exception as e:
            #traceback.print_exc()
            print e
        finally:
            try:
                p.selectedDataSource.resultsSource.close()
            except:
                pass

