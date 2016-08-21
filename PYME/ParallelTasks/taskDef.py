#!/usr/bin/python

##################
# taskDef.py
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

import random

class Task:
    def __init__(self, resultsURI=None):
        self.taskID = repr(random.random())
        self.timeout = 300 #seconds
        self.resultsURI = resultsURI
    def initializeWorkerTimeout(self, curtime):
        self.workerTimeout = curtime + self.timeout

class TaskResult:
    def __init__(self, task):
        self.taskID = task.taskID
        if 'queueID' in dir(task):
            self.queueID = task.queueID

class myTask(Task):
    def __init__(self):
        Task.__init__(self)
    def __call__(self):
        print("Hello")
