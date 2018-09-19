
import sys
import os
import errno
import glob
import time
import datetime
import traceback

import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

import PYME.experimental.dcimgSpoolShim as DCIMGSpool
from PYME.IO import PZFFormat

def safe_remove(path):
    """
    File removal with error handling. Defining function to clean up the FileChucker code
    
    Parameters
       ----------
       path : str
           The fully resolved filename of the file to be deleted
       Returns
       -------
       
    """
    try:
        os.remove(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

class DumpsterRaccoon(object):
    """

    The DumpsterRaccoon does keeps track of files we try to delete before a non-PYME acquisition software releases the
    file. The raccoon has methods to retry deletion, but is currently not run in its own thread, and therefor these
    methods are called explicitly by the FileChucker.

    """
    def __init__(self, num_retries=3):
        """

        Parameters
        ----------
        num_retries : int
            Maximum number of attempts to be made in deleting any given file before ignoring it

        """
        self.retries = num_retries
        self.dumpster = {}

    def __len__(self):
        """

        Returns
        -------
        num_paths : int
            The number of files which are currently queued to be removed

        """
        return len(self.dumpster.keys())

    def hoard_morsel(self, path):
        """

        Increments the number of attempts that have been made to remove a file and handles deletion of that file from
        the queue if self.retries is exceeded.

        Parameters
        ----------
        path : str
            Full file path to the file to be garbage-collected

        Returns
        -------

        """
        try:  # if Raccoon already knows about the trash, increment the number of times he's tried to eat it
            self.dumpster[path] += 1
            if self.dumpster[path] < self.retries:
                self.dumpster[path] += 1
            else:  # this one's just not going down, forget about it
                del self.dumpster[path]

        except KeyError:  # this is the first time Raccoon has seen this trash, store for later consumption
            self.dumpster[path] = 0

    def eat_trash(self, path):
        """

        Try to delete a single file. If it fails, hoard the tasty morsel for later consumption.

        Parameters
        ----------
        path : str
            Full file path to the file to be garbage-collected

        Returns
        -------

        """
        try:
            safe_remove(path)
            # if we're successful, make sure to remove this morsel from our memory
            self.dumpster.pop(path, None)
        except Exception:
            self.hoard_morsel(path)
            logger.exception(traceback.format_exc())

    def skim_trash(self):
        """

        Try to remove a single piece of trash from the queue.

        Returns
        -------

        """
        if len(self) > 0:
            self.eat_trash(self.dumpster.keys()[0])

    def dumpster_fire(self):
        """

        Try to remove each piece of trash currently in the queue.

        Returns
        -------

        """
        for k in self.dumpster.keys():
            self.eat_trash(k)



class venerableFileChucker(object):
    """
    fileChucker searches a given folder and hucks the DCIMG files it finds there onto the cluster.
    Once started, it does not stop unless slain by the user.
    This is certainly not the most elegant way of implementing the DCIMGSpooler, but may suffice for now....

    """
    def __init__(self, searchFolder, timeout=12, quantize=False):
        """

        Parameters
        ----------
        searchFolder : str
            Path the the folder to watch
        timeout : float
            time after which a series is deemed to be dead, in s. Default is 1 hr.
        """
        self.folder = searchFolder

        self.spooler = DCIMGSpool.DCIMGSpoolShim(timeout=timeout)
        self.timeout = timeout
        self.comp_settings = {'quantization': PZFFormat.DATA_QUANT_SQRT if quantize else PZFFormat.DATA_QUANT_NONE}

        self.dumpster_raccoon = DumpsterRaccoon()


    def _spoolSeries(self, mdfilename, delete_after_spool=False):
        """
        Spool a single series, which already exists

        Parameters
        ----------
        mdfilename : str
            The fully resolved filename of the series metadata

        Returns
        -------

        """
        self.spooler.OnNewSeries(mdfilename, self.comp_settings)
        series_stub =  mdfilename.strip('.json')
        print(datetime.datetime.utcnow())

        # find chunks for this metadata file (this should return the full paths)
        chunkList = sorted(glob.glob(series_stub + '*.dcimg'))

        for chunk in chunkList:
            try:
                self.spooler.OnDCIMGChunkDetected(chunk)
                #time.sleep(.5)   # FIXME: When using HUFFMANCODE more frames are lost without waiting
                #wait until we've sent everything
                #this is a bit of a hack
                #time.sleep(.1)
                while not self.spooler.spooler.finished():
                    time.sleep(.1)
            finally:
                if delete_after_spool:
                    self.dumpster_raccoon.eat_trash(chunk)

        # spool _events.json
        events_filename = series_stub + '_events.json' #note that we ignore this at present, kept for future compatibility
        zsteps_filename = series_stub + '_zsteps.json'

        try:
            self.spooler.OnSeriesComplete(events_filename, zsteps_filename, pushTaskToCluster=True)

            print(datetime.datetime.utcnow())
            # TODO: Add Feedback from cluster and also speed up writing in cluster
            # time.sleep(10)
        finally:
            if delete_after_spool:
                self.dumpster_raccoon.dumpster_fire()
                self.dumpster_raccoon.eat_trash(mdfilename)
                self.dumpster_raccoon.eat_trash(events_filename)
                self.dumpster_raccoon.eat_trash(zsteps_filename)

    def huck(self, mdFile, delete_after_spool, only_spool_complete):
        """
        Spools a series to the cluster
        Parameters
        ----------
        mdFile : str
            Fully resolved path to metadata file for series to be spooled
        delete_after_spool : bool
            Flag to determine whether files are deleted after spooling (True) or spooled without deletion (False)
        only_spool_complete : bool
            Flag to determine whether dcimg files should be spooled as soon as they are written (False), i.e. before an
            events file is written, or whether the FileChucker should wait until all dcimg files for a series are
            available before spooling (True).

        Returns
        -------

        """
        # compute the whole series name
        mdfilename = os.path.join(self.folder, mdFile)
        self.spooler.OnNewSeries(mdfilename)

        # calculate the stub with which all files start
        series_stub = mdfilename.strip('.json')

        # calculate event and zsteps filenames from stub
        events_filename = series_stub + '_events.json'
        zsteps_filename = series_stub + '_zsteps.json'

        # which chunks from this series have we already spooled? We do not want to spool them again
        spooled_chunks = []

        spooling_complete = False
        spool_timeout = time.time() + self.timeout
        while (not spooling_complete) and (time.time() < spool_timeout):
            # check to see if we have an events file (our end signal)
            on_disk = (os.path.exists(events_filename) or os.path.exists(zsteps_filename))
            if on_disk:
                # flag as complete so we exit the loop after spooling
                spooling_complete = True
            if only_spool_complete and (not spooling_complete):
                # keep waiting
                time.sleep(0.1)
                continue

            # find chunks for this metadata file (this should return the full paths)
            chunkList = sorted(glob.glob(series_stub + '*.dcimg'))

            for chunk in [c for c in chunkList if c not in spooled_chunks]:
                try:
                    spooled_chunks.append(chunk)
                    self.spooler.OnDCIMGChunkDetected(chunk)
                    # time.sleep(.5)  # FIXME: When using HUFFMANCODE more frames are lost without waiting
                    # wait until we've sent everything
                    # this is a bit of a hack
                    while not self.spooler.spooler.finished():
                        time.sleep(.1)
                finally:
                    if delete_after_spool:
                        # TODO: update this to only delete files if they are sent successfully
                        self.dumpster_raccoon.eat_trash(chunk)



        # we have seen our events file, the current series is complete
        try:
            self.spooler.OnSeriesComplete(events_filename, zsteps_filename, pushTaskToCluster=True)
            logger.debug('Finished spooling series %s' % series_stub)
        finally:
            if delete_after_spool:
                self.dumpster_raccoon.dumpster_fire()
                self.dumpster_raccoon.eat_trash(mdfilename)
                self.dumpster_raccoon.eat_trash(events_filename)
                self.dumpster_raccoon.eat_trash(zsteps_filename)

    def searchAndHuck(self, only_spool_new=False, delete_after_spool=False, only_spool_complete=False):
        """
        Monitors a folder (self.folder) and spool series which are written there and saved as dcimg files to the cluster

        Parameters
        ----------
        only_spool_new : bool
            Flag to determine whether the FileChucker should ignore existing files (True) or spool them (False).
        delete_after_spool : bool
            Flag to determine whether files are deleted after spooling (True) or spooled without deletion (False)
        only_spool_complete : bool
            Flag to determine whether dcimg files should be spooled as soon as they are written (False), i.e. before an
            events file is written, or whether the FileChucker should wait until all dcimg files for a series are
            available before spooling (True).

        Returns
        -------

        """
        md_candidates = glob.glob(os.path.join(self.folder, '*.json'))
        #changed it to just use a list comprehension as this will be much easier to read (and there is no performance advantage to using arrays) - DB
        metadataFiles = [f for f in md_candidates if not (f.endswith('_events.json') or f.endswith('_zsteps.json'))]


        if not only_spool_new:
            for mdFile in metadataFiles:
                mdPath = os.path.join(self.folder, mdFile)
                try:
                    self._spoolSeries(mdPath, delete_after_spool=delete_after_spool)
                except Exception:
                    logger.exception(traceback.format_exc())
                except KeyboardInterrupt:
                    sys.exit()

        #ignore metadata files corresponding to series we have already spooled
        ignoreList = set(metadataFiles)

        while True: #NB!!!!: this will run for ever
            # search for new files, use sets for sake of speed
            md_candidates = set(glob.glob(self.folder + '\*.json'))
            # keep ignoreList from growing unnecessarily; only track files that still exist
            ignoreList &= md_candidates
            # generate set of only the basic metadata files
            chaff = [f for f in md_candidates if (f.endswith('_events.json') or f.endswith('_zsteps.json'))]
            metadataFiles = md_candidates.difference(chaff).difference(ignoreList)
            try:
                # check if there are any series to spool
                if len(metadataFiles) < 1:
                    raise IndexError

                # add files about to be spooled to the ignore list
                ignoreList |= metadataFiles
                # spool series
                for mdFile in metadataFiles:
                    self.huck(mdFile, delete_after_spool, only_spool_complete)

            except IndexError:
                #this happens if there are no new metadata files - wait until there are some.
                time.sleep(.1) #wait just long enough to stop us from being in a CPU busy loop
                pass
            except KeyboardInterrupt:
                sys.exit() #make sure other threads get killed if we kill with ctrl-c
            except Exception:
                logger.exception(traceback.format_exc())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='only_spool_new', action='store_true',
                        help='Only spool new files as they are saved')
    parser.add_argument('-d', dest='delete_after_spool', action='store_true',
                        help='Delete files after they are spooled to the cluster')
    parser.add_argument('-c', dest='only_spool_complete', action='store_true',
                        help='spool series as dcimg files are written, rather than waiting for the entire series')
    parser.add_argument('-q', dest='quantize', action='store_true',
                        help='Quantize with sqrt(N) interval scaling')
    parser.add_argument('testFolder', metavar='testFolder', type=str,
                        help='Folder for fileChucker to monitor')
    parser.add_argument('-t', '--timeout', type=float, default=12.,
                        help='Time to wait for single series to spool')
    args = parser.parse_args()

    searcher = venerableFileChucker(args.testFolder, timeout=args.timeout, quantize=args.quantize)
    searcher.searchAndHuck(args.only_spool_new, args.delete_after_spool, args.only_spool_complete)
