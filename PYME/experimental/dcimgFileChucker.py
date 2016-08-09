

import os
import numpy as np
import PYME.experimental.dcimgSpoolShim as DCIMGSpool

class venerableFileChucker:
    """
    fileChucker searches a given folder and hucks the DCIMG files it finds there onto the cluster.
    Once started, it does not stop unless slain by the user.
    This is certainly not the most elegant way of implementing the DCIMGSpooler, but may suffice for now....

    """
    def __init__(self, searchFolder):
        self.folder = searchFolder
        self.ignoreList = []

        self.spooler = DCIMGSpool.DCIMGSpoolShim()

    def searchAndHuck(self, onlySpoolNew=None, delAfterSpool=None):
        self.flist = np.array(os.listdir(self.folder))
        try:
            mdList = self.flist[np.where(np.bitwise_and([fileName.endswith('.json') for fileName in self.flist],
                                np.invert([fileName.endswith('_events.json') for fileName in self.flist])))]
        except:
            mdList = []

        if not onlySpoolNew:

            for mdFile in mdList:
                mdPath = os.path.join(self.folder, mdFile)
                self.spooler.OnNewSeries(mdPath)
                # find chunks for this mdFile
                chunkList = self.flist[np.where(np.bitwise_and([fileName.endswith('.dcimg') for fileName in self.flist],
                                [fileName.startswith(mdFile.strip('.json')) for fileName in self.flist]))]
                for chunk in chunkList:
                    chunkPath = os.path.join(self.folder, chunk)
                    self.spooler.OnDCIMGChunkDetected(chunkPath)
                    if delAfterSpool:
                        os.remove(chunkPath)
                # spool events.json
                events = self.flist[np.where(np.bitwise_and([fileName.endswith('_events.json') for fileName in self.flist],
                                   [fileName.startswith(mdFile.strip('.json')) for fileName in self.flist]))]
                eventsPath = os.path.join(self.folder, events[0])
                self.spooler.OnSeriesComplete(eventsPath)
                # TODO: Add Feedback from cluster and also speed up writing in cluster
                # time.sleep(10)              
                if delAfterSpool:
                    os.remove(mdPath)
                    os.remove(eventsPath)

        ignoreList = mdList

        while True:
            # search for new files
            self.flist = np.array(os.listdir(self.folder))

            try:
                mdList = self.flist[np.where(np.bitwise_and([fileName.endswith('.json') for fileName in self.flist],
                                np.invert([fileName.endswith('_events.json') for fileName in self.flist])))]
                mdList = list(set(mdList).difference(set(ignoreList)))
                mdList.sort(reverse=True)
                mdFile = mdList[0]
                mdPath = os.path.join(self.folder, mdFile)
                self.spooler.OnNewSeries(mdPath)
                breaker = None
                while not breaker:
                    # find chunks for this mdFile
                    self.flist = np.array(os.listdir(self.folder))
                    chunkList = self.flist[np.where(np.bitwise_and([fileName.endswith('.dcimg') for fileName in self.flist],
                                [fileName.startswith(mdFile.strip('.json')) for fileName in self.flist]))]

                    events = self.flist[np.where(np.bitwise_and([fileName.endswith('_events.json') for fileName in self.flist],
                                [fileName.startswith(mdFile.strip('.json')) for fileName in self.flist]))]

                    try:
                        chunkPath = os.path.join(self.folder, chunkList[0])
                        self.spooler.OnDCIMGChunkDetected(chunkPath)
                        # time.sleep(1)
                        if delAfterSpool:
                            # TODO: update this to only delete files if they are sent successfully
                            os.remove(chunkPath)

                    except IndexError:
                        # this means we have sent off all chunks. will still raise other errors
                        try:
                            # Check if events file is ready
                            events_log = events[0]
                            breaker = True
                            eventsPath = os.path.join(self.folder, events_log)
                            self.spooler.OnSeriesComplete(eventsPath)
                            if delAfterSpool:
                                os.remove(mdPath)
                                os.remove(eventsPath)
                        except IndexError:
                            pass

            except:
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
