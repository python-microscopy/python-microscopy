

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

                # TODO: Add Feedback from cluster and also speed up writing in cluster
                # time.sleep(10)
                self.spooler.OnSeriesComplete()
                if delAfterSpool:
                    os.remove(mdPath)


        ignoreList = mdList

        while True:
            # search for new files
            self.flist = np.array(os.listdir(self.folder))

            try:
                mdList = self.flist[np.where(np.bitwise_and([fileName.endswith('.json') for fileName in self.flist],
                                np.invert([fileName.endswith('_events.json') for fileName in self.flist])))]
                mdFile = list(set(mdList).difference(set(ignoreList)))[0]
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
                            os.remove(os.path.join(chunkPath))

                    except IndexError:
                        # this means we have sent off all chunks. will still raise other errors
                        try:
                            # Check if events file is ready
                            events_log = events[0]
                            breaker = True
                            if delAfterSpool:
                                os.remove(os.path.join(self.folder, mdFile))
                                os.remove(os.path.join(self.folder, events_log))
                        except IndexError:
                            pass


                self.spooler.OnSeriesComplete()
                if delAfterSpool:
                    os.remove(mdPath)

            except:
                pass


if __name__ == "__main__":
    #TODO: add testfolder, spoolOnlyNew, and deleteAfterSpool as command line arguments
    testFolder = 'C:\PYMEtestData\\2016_6_15_DIMCG_with_JSON'
    searcher = venerableFileChucker(testFolder)
    # searcher.searchAndHuck(onlySpoolNew=True, delAfterSpool=None)
    searcher.searchAndHuck()