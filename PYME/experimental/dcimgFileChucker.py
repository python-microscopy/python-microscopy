

import os
import glob
import numpy as np
import time

import PYME.experimental.dcimgSpoolShim as DCIMGSpool

class venerableFileChucker:
    """
    fileChucker searches a given folder and hucks the DCIMG files it finds there onto the cluster.
    Once started, it does not stop unless slain by the user.
    This is certainly not the most elegant way of implementing the DCIMGSpooler, but may suffice for now....

    """
    def __init__(self, searchFolder, timeout = 3600):
        """

        Parameters
        ----------
        searchFolder : str
            Path the the folder to watch
        timeout : float
            time after which a series is deemed to be dead, in s. Default is 1 hr.
        """
        self.folder = searchFolder
        self.ignoreList = []

        self.spooler = DCIMGSpool.DCIMGSpoolShim()
        self.timeout = timeout

    def _spoolSeries(self, mdfilename, deleteAfterSpool=False):
        """
        Spool a single series, which already exists

        Parameters
        ----------
        mdfilename : str
            The fully resolved filename of the series metadata

        Returns
        -------

        """
        self.spooler.OnNewSeries(mdfilename)
        series_stub =  mdfilename.strip('.json')

        # find chunks for this metadata file (this should return the full paths)
        chunkList = glob.glob(series_stub + '*.dcimg') #TODO - should we sort this?

        for chunk in chunkList:
            self.spooler.OnDCIMGChunkDetected(chunk)
            if deleteAfterSpool:
                os.remove(chunk)

        # spool _events.json
        events_filename = series_stub + '_events.json' #note that we ignore this at present, kept for future compatibility
        zsteps_filename = series_stub + '_zsteps.json'

        self.spooler.OnSeriesComplete(events_filename, zsteps_filename)

        # TODO: Add Feedback from cluster and also speed up writing in cluster
        # time.sleep(10)
        if deleteAfterSpool:
            os.remove(mdfilename)
            os.remove(zsteps_filename)
            #os.remove(events_filename)

    def searchAndHuck(self, onlySpoolNew=False, deleteAfterSpool=False):
        md_candidates = glob.glob(self.folder + '*.json')
        #changed it to just use a list comprehension as this will be much easier to read (and there is no performance advantage to using arrays) - DB
        metadataFiles = [f for f in md_candidates if not (f.endswith('_events.json') or f.endswith('_zsteps.json'))]


        if not onlySpoolNew:
            for mdFile in metadataFiles:
                mdPath = os.path.join(self.folder, mdFile)
                self._spoolSeries(mdPath, deleteAfterSpool=deleteAfterSpool)

        #ignore metadata files corresponding to series we have already spooled
        ignoreList = metadataFiles

        while True: #NB!!!!: this will run for ever
            # search for new files
            md_candidates = glob.glob(self.folder + '*.json')

            try:
                metadataFiles = [f for f in md_candidates if
                                 not (f in ignoreList or f.endswith('_events.json') or f.endswith('_zsteps.json'))]
                metadataFiles.sort(reverse=True)

                #get the oldest metadata file not on our list of files to be ignored
                mdFile = metadataFiles[0]

                #compute the whole series name
                mdfilename = os.path.join(self.folder, mdFile)
                self.spooler.OnNewSeries(mdfilename)

                #calculate the stub with which all files start
                series_stub = mdfilename.strip('.json')

                #calculate event and zsteps filenames from stub
                events_filename = series_stub + '_events.json'
                zsteps_filename = series_stub + '_zsteps.json'

                #which chunks from this series have we already spooled? We don;t want to spool them again
                spooled_chunks = []

                events_file_detected = False
                spool_timeout = time.time() + self.timeout
                while (not events_file_detected) and (time.time() < spool_timeout):
                    # check to see if we have an events file (our end signal)
                    events_file_detected = os.path.exists(events_filename)

                    # find chunks for this metadata file (this should return the full paths)
                    chunkList = glob.glob(series_stub + '*.dcimg') #TODO - should we sort this?

                    for chunk in chunkList:
                        if not chunk in spooled_chunks:
                            self.spooler.OnDCIMGChunkDetected(chunk)
                            # time.sleep(1)
                            if deleteAfterSpool:
                                # TODO: update this to only delete files if they are sent successfully
                                os.remove(chunk)

                #we have seen our events file, the current series is complete
                self.spooler.OnSeriesComplete(events_filename, zsteps_filename)
                self.ignoreList.append(mdfilename)
                if deleteAfterSpool:
                    os.remove(mdfilename)
                    os.remove(events_file_detected)

            except IndexError:
                #this happens if there are no new metadata files - wait until there are some.
                time.sleep(.1) #wait just long enough to stop us from being in a CPU busy loop
                pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='onlySpoolNew', action='store_true',
                        help='Only spool new files as they are saved')
    parser.add_argument('-d', dest='delAfterSpool', action='store_true',
                        help='Delete files after they are spooled to the cluster')
    parser.add_argument('testFolder', metavar='testFolder', type=str,
                        help='Folder for fileChucker to monitor')
    args = parser.parse_args()

    searcher = venerableFileChucker(args.testFolder)
    searcher.searchAndHuck(args.onlySpoolNew, args.delAfterSpool)
