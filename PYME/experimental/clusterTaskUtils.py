

def _verifyClusterResultsFilename(resultsFilename):
    from PYME.IO import clusterIO
    import os
    if clusterIO.exists(resultsFilename):
        di, fn = os.path.split(resultsFilename)
        i = 1
        stub = os.path.splitext(fn)[0]
        while clusterIO.exists(os.path.join(di, stub + '_%d.h5r' % i)):
            i += 1

        resultsFilename = os.path.join(di, stub + '_%d.h5r' % i)

    return resultsFilename


def _launch_localize(analysisMDH, seriesName):
    import logging
    import json
    from PYME.ParallelTasks import HTTPTaskPusher
    from PYME.IO import MetaDataHandler
    from PYME.Analysis import MetaData
    from PYME.IO.FileUtils.nameUtils import genClusterResultFileName
    from PYME.IO import unifiedIO

    resultsFilename = _verifyClusterResultsFilename(genClusterResultFileName(seriesName))
    logging.debug('Results file: ' + resultsFilename)

    resultsMdh = MetaDataHandler.NestedClassMDHandler(analysisMDH)
    resultsMdh.update(json.loads(unifiedIO.read(seriesName + '/metadata.json')))

    resultsMdh['EstimatedLaserOnFrameNo'] = resultsMdh.getOrDefault('EstimatedLaserOnFrameNo', resultsMdh.getOrDefault('Analysis.StartAt', 0))
    MetaData.fixEMGain(resultsMdh)
    #resultsMdh['DataFileID'] = fileID.genDataSourceID(image.dataSource)

    #TODO - do we need to keep track of the pushers in some way (we currently rely on the fact that the pushing thread
    #will hold a reference
    pusher = HTTPTaskPusher.HTTPTaskPusher(dataSourceID=seriesName,
                                                metadata=resultsMdh, resultsFilename=resultsFilename)

    logging.debug('Queue created')