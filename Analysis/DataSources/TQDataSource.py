class DataSource:
    moduleName = 'TQDataSource'
    def __init__(self, queueName, taskQueue):
        self.taskQueue = taskQueue
        self.queueName = queueName

    def getSlice(self, ind):
        return self.taskQueue.getQueueData(self.queueName, 'ImageData',ind)

    def getSliceShape(self):
        return self.taskQueue.getQueueData(self.queueName, 'ImageShape')

    def getNumSlices(self):
        return self.taskQueue.getQueueData(self.queueName, 'NumSlices')

    def getEvents(self):
        return self.taskQueue.getQueueData(self.queueName, 'Events')
