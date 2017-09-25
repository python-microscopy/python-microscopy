
import sys
import os
import errno
import glob
import time
import datetime

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

class venerableFileChucker(object):
    """
    fileChucker searches a given folder and hucks the DCIMG files it finds there onto the cluster.
    Once started, it does not stop unless slain by the user.
    This is certainly not the most elegant way of implementing the DCIMGSpooler, but may suffice for now....

    """
    def __init__(self, searchFolder, timeout = 12, quantize=False):
        """

        Parameters
        ----------
        searchFolder : str
            Path the the folder to watch
        timeout : float
            time after which a series is deemed to be dead, in s. Default is 1 hr.
        """
        self.folder = searchFolder

        self.spooler = DCIMGSpool.DCIMGSpoolShim()
        self.timeout = timeout
        self.comp_settings = {'quantization': PZFFormat.DATA_QUANT_SQRT if quantize else PZFFormat.DATA_QUANT_NONE}


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
            self.spooler.OnDCIMGChunkDetected(chunk)
            #time.sleep(.5)   # FIXME: When using HUFFMANCODE more frames are lost without waiting
            #wait until we've sent everything
            #this is a bit of a hack
            #time.sleep(.1)
            while not self.spooler.spooler.postQueue.empty() or (self.spooler.spooler.numThreadsProcessing > 0):
                time.sleep(.1)
            if delete_after_spool:
                safe_remove(chunk)

        # spool _events.json
        events_filename = series_stub + '_events.json' #note that we ignore this at present, kept for future compatibility
        zsteps_filename = series_stub + '_zsteps.json'

        self.spooler.OnSeriesComplete(events_filename, zsteps_filename, pushTaskToCluster=True)

        print(datetime.datetime.utcnow())
        # TODO: Add Feedback from cluster and also speed up writing in cluster
        # time.sleep(10)
        if delete_after_spool:
            safe_remove(mdfilename)
            safe_remove(events_filename)
            safe_remove(zsteps_filename)

    def huck(self, mdFile, delete_after_spool, only_spool_complete):
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
            if only_spool_complete and not on_disk:
                # keep waiting
                time.sleep(0.1)
                continue
            else:
                # flag as complete so we exit the loop after spooling
                spooling_complete = True

            # find chunks for this metadata file (this should return the full paths)
            chunkList = sorted(glob.glob(series_stub + '*.dcimg'))

            for chunk in [c for c in chunkList if c not in spooled_chunks]:
                self.spooler.OnDCIMGChunkDetected(chunk)
                # time.sleep(.5)  # FIXME: When using HUFFMANCODE more frames are lost without waiting
                # wait until we've sent everything
                # this is a bit of a hack
                while not self.spooler.spooler.postQueue.empty() or (self.spooler.spooler.numThreadsProcessing > 0):
                    time.sleep(.1)
                if delete_after_spool:
                    # TODO: update this to only delete files if they are sent successfully
                    safe_remove(chunk)



        # we have seen our events file, the current series is complete
        self.spooler.OnSeriesComplete(events_filename, zsteps_filename, pushTaskToCluster=True)
        logger.debug('Finished spooling series %s' % series_stub)

        if delete_after_spool:
            safe_remove(mdfilename)
            safe_remove(events_filename)
            safe_remove(zsteps_filename)

    def searchAndHuck(self, only_spool_new=False, delete_after_spool=False, only_spool_complete=False):
        md_candidates = glob.glob(os.path.join(self.folder, '*.json'))
        #changed it to just use a list comprehension as this will be much easier to read (and there is no performance advantage to using arrays) - DB
        metadataFiles = [f for f in md_candidates if not (f.endswith('_events.json') or f.endswith('_zsteps.json'))]


        if not only_spool_new:
            for mdFile in metadataFiles:
                mdPath = os.path.join(self.folder, mdFile)
                try:
                    self._spoolSeries(mdPath, delete_after_spool=delete_after_spool)
                except Exception:
                    import traceback
                    logger.exception(traceback.format_exc())
                except KeyboardInterrupt:
                    sys.exit()

        #ignore metadata files corresponding to series we have already spooled
        ignoreList = set(metadataFiles)

        while True: #NB!!!!: this will run for ever
            # search for new files
            md_candidates = set(glob.glob(self.folder + '\*.json'))
            ignoreList &= md_candidates  # keep ignoreList from growing unnecessarily; only track files that still exist
            chaff = [f for f in md_candidates if (f.endswith('_events.json') or f.endswith('_zsteps.json'))]#glob.glob(self.folder + '\*_events.json')
            metadataFiles = md_candidates.difference(chaff).difference(ignoreList)
            try:
                #metadataFiles = [f for f in md_candidates if
                #                 not (f in ignoreList or f.endswith('_events.json') or f.endswith('_zsteps.json'))]
                #metadataFiles.sort()

                #get the oldest metadata file not on our list of files to be ignored
                # mdFile = metadataFiles[0]
                if len(metadataFiles) < 1:
                    raise IndexError
                #try:
                ignoreList |= metadataFiles  # add files about to be spooled to the ignore list
                for mdFile in metadataFiles:
                    #t = threading.Thread(target=self.huck, args=(mdFile, delete_after_spool, only_spool_complete))
                    #logging.debug('%s assigned %s' % (t.name, mdFile))
                    #t.start()

                #spool the file in a new thread and keep looking for new files
                #this stops us from freezing for an hour waiting for a file to complete if something goes wrong with labview  # NB: we already had/have a timeout - AB
                #t = threading.Thread(target=self.huck, args=(mdFile, delete_after_spool, only_spool_complete))
                #t.start()

                    self.huck(mdFile, delete_after_spool, only_spool_complete)
                #except KeyboardInterrupt:
                    #allow us to interrupt with ctrl-c
                #    raise
                #except Exception:
                    # log the error and continue - this is FileChucker is venerable, and shant be stopped easily
                #    import traceback
                #    logger.exception(traceback.format_exc())

                #ignoreList.append(mdFile)

            except IndexError:
                #this happens if there are no new metadata files - wait until there are some.
                time.sleep(.1) #wait just long enough to stop us from being in a CPU busy loop
                pass
            except KeyboardInterrupt:
                sys.exit() #make sure other threads get killed if we kill with ctrl-c
            except Exception:
                import traceback
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
    args = parser.parse_args()

    searcher = venerableFileChucker(args.testFolder, quantize=args.quantize)
    searcher.searchAndHuck(args.only_spool_new, args.delete_after_spool, args.only_spool_complete)
