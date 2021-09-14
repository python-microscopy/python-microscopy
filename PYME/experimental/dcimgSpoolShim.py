import json
import os
import time

from PYME.IO import HTTPSpooler
from PYME.Analysis import MetaData
from PYME.IO import MetaDataHandler, clusterIO, unifiedIO
from PYME.IO.DataSources import DcimgDataSource, MultiviewDataSource
from PYME.IO.clusterExport import ImageFrameSource, MDSource
from PYME.cluster import HTTPRulePusher

# DT is the period of time the spooler will wait before checking if a file is free to be spooled (i.e. external
# acquisition software has finished writing the dcimg file)
DT = 0.1

def _writing_finished(filename):
    """Check to see whether anyone else has this file open.

    the check is performed by attempting to rename the file"""

    try:
        #rename the file and then rename back. This will fail if
        #another process has the file open
        os.rename(filename, filename + '_')
        os.rename(filename + '_', filename)
        return True
    except:
        return False

def _wait_for_file(filename, timeout=20):
    """
    Wait until others have finished with the file.

    WARNING/FIXME: This can block for ever. Set the up limit of waiting to 20 seconds by default

    Parameters
    ----------
    filename : str
        The fully resolved path of the file in question
    timeout : int or float
        Time period to keep checking whether the file is written before giving up

    Returns
    -------
    finished : bool
        True if file is no longer held by external programs, False if still in use by other programs after the timeout.

    """
    stop = time.time() + timeout
    while not _writing_finished(filename):
        time.sleep(DT)
        if time.time() >= stop:
            return False
    return True


class DCIMGSpoolShim(object):
    """
    DCIMGSpoolShim provides methods to interface between DcimgDataSource and HTTPSpooler, so that one can spool
    dcimg files (containing arbitary numbers of image frames) as they are finished writing.
    """
    def __init__(self, timeout):
        self.n_spooled = 0
        self.timeout = timeout
    
    def OnNewSeries(self, metadataFilename, comp_settings=None):
        """Called when a new series is detected (ie the <seriesname>.json)
        file is detected
        """
        if comp_settings is None:
            comp_settings = {}
        # Make sure that json file is done writing
        success = _wait_for_file(metadataFilename, self.timeout)
        if not success:
            raise UserWarning('dcimg file is taking too long to finish writing')


        #create an empty metadatahandler
        self.mdh = MetaDataHandler.NestedClassMDHandler(MetaData.BareBones)

        #load metadata from file and insert into our metadata handler
        with open(metadataFilename, 'r') as f:
            mdd = json.load(f)
            self.mdh.update(mdd)

        #determine a filename on the cluster from our local filename
        #TODO - make this more complex to generate suitable directory structures
        filename = unifiedIO.verbose_fix_name(os.path.splitext(metadataFilename)[0])
        
        dirname, seriesname = os.path.split(filename)
        
        #Strip G:\\ in filename to test if it caused connection problem to some nodes in cluster
        dirname = dirname[dirname.find('\\') + 1:]
        
        cluster_filename = os.path.join(dirname, '%03d' % int(self.n_spooled/1000), seriesname)
        
        #create virtual frame and metadata sources
        self.imgSource = ImageFrameSource()
        self.metadataSource = MDSource(self.mdh)
        MetaDataHandler.provideStartMetadata.append(self.metadataSource)

        #generate the spooler
        comp_settings.update({'quantizationOffset': self.mdh.getOrDefault('Camera.ADOffset', 0)})
        self.spooler = HTTPSpooler.Spooler(cluster_filename, self.imgSource.onFrame, frameShape=None,
                                           compressionSettings=comp_settings)

        self.n_spooled += 1
        #spool our data
        self.spooler.StartSpool()

    def OnDCIMGChunkDetected(self, chunkFilename):
        """Called whenever a new chunk is detected.
        spools that chunk to the cluster"""
        success = _wait_for_file(chunkFilename, self.timeout)
        if not success:
            raise UserWarning('dcimg file is taking too long to finish writing')

        chunk = DcimgDataSource.DataSource(chunkFilename)
        croppedChunk = MultiviewDataSource.DataSource(chunk, self.mdh)

        self.imgSource.spoolData(croppedChunk)
        self.spooler.FlushBuffer()

    def OnSeriesComplete(self, eventsFilename=None, zstepsFilename=None, pushTaskToCluster=False):
        """Called when the series is finished (ie we have seen)
        the events file"""

        if (not eventsFilename is None) and (os.path.exists(eventsFilename)):
            # Update event Log with events.json
            with open(eventsFilename, 'r') as f:
                 events = json.load(f)

            for evt in events:
                name, descr, timestamp = evt
                self.spooler.evtLogger.logEvent(eventName=name, eventDescr=descr, timestamp=float(timestamp))

        if (not zstepsFilename is None) and (os.path.exists(zstepsFilename)):
            #create pseudo events based on our zstep information
            with open(zstepsFilename, 'r') as f:
                zsteps = json.load(f)

            positions = zsteps['PIFOC_positions']
            startFrames = zsteps['Start_Frame_eachZ']

            startTime = self.mdh.getOrDefault('StartTime', 0)
            cycleTime = self.mdh.getOrDefault('Camera.CycleTime', 0.01) #use a default frame length of 10 ms. Not super critical

            for pos, fr in zip(positions, startFrames):
                fakeTime = startTime + cycleTime*fr
                self.spooler.evtLogger.logEvent(eventName='StartAq', eventDescr='%d' % fr, timestamp=fakeTime)
                self.spooler.evtLogger.logEvent(eventName='ProtocolFocus', eventDescr='%d, %3.3f' % (fr, pos),
                                                timestamp=fakeTime)
        
        self.spooler.StopSpool()
        self.spooler.FlushBuffer()

        if pushTaskToCluster:

            self.mdh.setEntry('Analysis.BGRange', [-32, 0])
            self.mdh.setEntry('Analysis.DebounceRadius', 4)
            self.mdh.setEntry('Analysis.DetectionThreshold', 0.75)
            self.mdh.setEntry('Analysis.FiducialThreshold', 1.8)
            self.mdh.setEntry('Analysis.FitModule', 'AstigGaussGPUFitFR')
            self.mdh.setEntry('Analysis.PCTBackground', 0.25)
            self.mdh.setEntry('Analysis.ROISize', 7.5)
            self.mdh.setEntry('Analysis.StartAt', 32)
            self.mdh.setEntry('Analysis.TrackFiducials', False)
            self.mdh.setEntry('Analysis.subtractBackground', True)
            self.mdh.setEntry('Analysis.GPUPCTBackground', True)
            cluster_filename = 'pyme-cluster://%s/%s' % (clusterIO.local_serverfilter, self.spooler.seriesName)
            HTTPRulePusher.launch_localize(analysisMDH=self.mdh, seriesName=cluster_filename)

        #remove the metadata generator
        MetaDataHandler.provideStartMetadata.remove(self.metadataSource)
