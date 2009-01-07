class remoteDataSource:
    def __init__(self, taskQueue, queueName):
        self.taskQueue = taskQueue
        self.queueName = queueName

    def getSlice(self, ind):
        return self.taskQueue.getQueueData(self.QueueName, 'ImageData',ind)

    def getSliceShape(self):
        return self.taskQueue.getQueueData(self.QueueName, 'ImageShape')
