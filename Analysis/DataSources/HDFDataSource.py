from PYME.ParallelTasks.relativeFiles import getFullFilename
import tables

class DataSource:
    def __init__(self, h5Filename, taskQueue):
        self.h5Filename = getFullFilename(h5Filename)#convert relative path to full path
        self.h5File = tables.openFile(self.h5Filename)

    def getSlice(self, ind):
        if ind >= self.h5File.root.ImageData.shape[0]:
                self.reloadData() #try reloading the data in case it's grown
        
        return self.h5File.root.ImageData[ind, :,:]


    def getSliceShape(self):
        return self.h5File.root.ImageData.shape[1:3]

    def getNumSlices(self):
        return self.h5File.root.ImageData.shape[0]

    def release(self):
        self.h5File.close()

    def reloadData(self):
        self.h5File.close()
        self.h5File = tables.openFile(self.h5Filename)
