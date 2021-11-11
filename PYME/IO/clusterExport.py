# -*- coding: utf-8 -*-
"""
Created on Sun May 22 17:13:51 2016

@author: david
"""

from PYME.IO import HTTPSpooler_v2 as HTTPSpooler
from PYME.IO import MetaDataHandler

from PYME.contrib import dispatch

class ImageFrameSource(object):
    def __init__(self):
        #self.image = image
        
        self.onFrame = dispatch.Signal(['frameData'])
        self.spoolProgress = dispatch.Signal(['percent'])
        
    def spoolImageFromFile(self, filename):
        """Load an image file and then spool"""
        from PYME.IO import image
        
        self.spoolData(image.ImageStack(filename).data)
        
    def spoolData(self, data):
        """Extract frames from a data source.
        
        Parameters
        ----------
        
        data : PYME.IO.DataSources.DataSource object
            the data source. Needs to implement the getNumSlices() and getSlice()
            methods.
        """
        nFrames = data.getNumSlices()
        for i in range(nFrames):
            self.onFrame.send(self, frameData=data.getSlice(i))
            if (i % 3000) == 0:
                self.spoolProgress.send(self, percent=float(i)/nFrames)
                print('Spooling %d of %d frames' % (i, nFrames))
            
          

class MDSource(object):
    """Spoof a metadata source for the spooler"""
    def __init__(self, mdh):
        self.mdh = mdh

    def __call__(self, md_to_fill):
        md_to_fill.copyEntriesFrom(self.mdh)
         
         
def ExportImageToCluster(image, filename, progCallback=None):
    """Exports the given image to a file on the cluster
    
    Parameters
    ----------
    
    image : PYME.IO.image.ImageStack object
        the source image
    filename : string
        the filename on the cluster
        
    """
    
    #create virtual frame and metadata sources
    imgSource = ImageFrameSource()
    mds = MDSource(image.mdh)
    MetaDataHandler.provideStartMetadata.append(mds)
    
    if not progCallback is None:
        imgSource.spoolProgress.connect(progCallback)
    
    #queueName = getRelFilename(self.dirname + filename + '.h5')
    
    #generate the spooler
    spooler = HTTPSpooler.Spooler(filename, imgSource.onFrame, frameShape = image.data.shape[:2])
    
    #spool our data    
    spooler.StartSpool()
    imgSource.spoolData(image.data)
    spooler.FlushBuffer()
    spooler.StopSpool()
    
    #remove the metadata generator
    MetaDataHandler.provideStartMetadata.remove(mds)


SERIES_PATTERN = '%(day)d_%(month)d_series_%(counter)'

def _getFilenameSuggestion(dirname='', seriesname = SERIES_PATTERN):
    from PYME.IO.FileUtils import nameUtils
    from PYME.IO import clusterIO
    import os
    
    if dirname == '':   
        dirname = nameUtils.genClusterDataFilepath()
    else:
        dirname = dirname.split(nameUtils.getUsername())[-1]
        
        dir_parts = dirname.split(os.path.sep)
        if len(dirname) < 1 or len(dir_parts) > 3:
            #path is either too complex, or too easy - revert to default
            dirname = nameUtils.genClusterDataFilepath()
        else:
            dirname = nameUtils.getUsername() + '/'.join(dir_parts)
    
    #dirname = defDir % nameUtils.dateDict
    seriesStub = dirname + '/' + seriesname % nameUtils.dateDict

    seriesCounter = 0
    seriesName = seriesStub % {'counter' : nameUtils.numToAlpha(seriesCounter)}
        
    #try to find the next available serie name
    while clusterIO.exists(seriesName + '/'):
        seriesCounter +=1
        
        if '%(counter)' in seriesName:
            seriesName = seriesStub % {'counter' : nameUtils.numToAlpha(seriesCounter)}
        else:
            seriesName = seriesStub + '_' + nameUtils.numToAlpha(seriesCounter)
            
    return seriesName


def suggest_cluster_filename(image):
    import os
    if not image.filename is None:
        dirname, seriesname = os.path.split(image.filename)
    
        return _getFilenameSuggestion(dirname, seriesname)
    else:
        return _getFilenameSuggestion()

