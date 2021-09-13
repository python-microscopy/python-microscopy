#!/usr/bin/python

##################
# block_dec.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

#f = f3
#import tcluster
#import dec
#from scipy import *
from PYME.misc import taskDef


queueID = None
decObj = None


class decResult:
    def __init__(self, task, results, queueID):
        self.taskID = task.taskID
        self.blocknum = task.blocknum
        self.results = results
        self.queueID = queueID

class decTask(taskDef.Task):
    def __init__(self, queueName, block, blocknum, lamb=1e-2, num_iters=10):
        taskDef.Task.__init__(self)

        self.queueName = queueName
        self.block = block
        self.blocknum = blocknum
        self.lamb = lamb
        self.num_iters = num_iters

    def __call__(self, gui=False, taskQueue=None):
        global queueID, decObj

        if (not queueID == self.queueName) or (not decObj.shape == self.block.shape):
            decObj = taskQueue.getQueueData(self.queueName, 'dec')
            decObj.prep()
            queueID = self.queueName

        print((self.blocknum, self.lamb))

        res = decObj.deconv(self.block, self.lamb, self.num_iters)
        return decResult(self, res, queueID)


