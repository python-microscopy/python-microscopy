import glob
import json
import time
import os

from PYME.IO.clusterExport import ImageFrameSource, MDSource
from PYME.IO import MetaDataHandler
from PYME.IO.DataSources import DcimgDataSource, MultiviewDataSource
from PYME.Analysis import MetaData
from PYME.Acquire import HTTPSpooler

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

def _wait_for_file(filename):
    """Wait until others have finished with the file.

    WARNING/FIXME: This can block for ever. Set the up limit of waiting to 20 seconds
    """
    ind = 0
    while not _writing_finished(filename):
        time.sleep(.1)
        ind += 1
        if ind >= 200:
            return False
    return True


class DCIMGSpoolShim:
    """
    DCIMGSpoolShim provides methods to interface between DcimgDataSource and HTTPSpooler, so that one can spool
    dcimg files (containing arbitary numbers of image frames) as they are finished writing.
    """
    def OnNewSeries(self, metadataFilename):
        """Called when a new series is detected (ie the <seriesname>.json)
        file is detected
        """
        # Make sure that json file is done writing
        success = _wait_for_file(metadataFilename)
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
        filename = os.path.splitext(metadataFilename)[0]
        #Strip G:\\ in filename to test if it caused connection problem to some nodes in cluster
        filename = filename[filename.find('\\') + 1:]
        #create virtual frame and metadata sources
        self.imgSource = ImageFrameSource()
        self.metadataSource = MDSource(self.mdh)
        MetaDataHandler.provideStartMetadata.append(self.metadataSource)

        #generate the spooler
        self.spooler = HTTPSpooler.Spooler(filename, self.imgSource.onFrame, frameShape=None)

        #spool our data
        self.spooler.StartSpool()

    def OnDCIMGChunkDetected(self, chunkFilename):
        """Called whenever a new chunk is detected.
        spools that chunk to the cluster"""
        success = _wait_for_file(chunkFilename)
        if not success:
            raise UserWarning('dcimg file is taking too long to finish writing')

        chunk = DcimgDataSource.DataSource(chunkFilename)
        croppedChunk = MultiviewDataSource.DataSource(chunk, self.mdh)

        self.imgSource.spoolData(croppedChunk)
        self.spooler.FlushBuffer()

    def OnSeriesComplete(self, eventsFilename=None, zstepsFilename=None, pushTasksToCluster=False):
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

        if pushTasksToCluster:
            #from PYME.ParallelTasks import HTTPTaskPusher
            #pusher = HTTPTaskPusher.HTTPTaskPusher()
            from PYME.experimental import clusterTaskUtils

            self.mdh.setEntry('Analysis.BGRange', [-30, 0])
            self.mdh.setEntry('Analysis.DebounceRadius', 4)
            self.mdh.setEntry('Analysis.DetectionThreshold', 0.8)
            self.mdh.setEntry('Analysis.FiducialThreshold', 1.8)
            self.mdh.setEntry('Analysis.FitModule', 'AstigGaussGPUFitFR')
            self.mdh.setEntry('Analysis.PCTBackground', 0.0)
            self.mdh.setEntry('Analysis.ROISize', 7.5)
            self.mdh.setEntry('Analysis.StartAt', 30)
            self.mdh.setEntry('Analysis.TrackFiducials', False)
            self.mdh.setEntry('Analysis.subtractBackground', True)
            clusterTaskUtils._launch_localize(analysisMDH=self.mdh, seriesName=self.spooler.seriesName)

        #remove the metadata generator
        MetaDataHandler.provideStartMetadata.remove(self.metadataSource)
