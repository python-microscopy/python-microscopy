#!/usr/bin/python

##################
# TQDataSource.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

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
