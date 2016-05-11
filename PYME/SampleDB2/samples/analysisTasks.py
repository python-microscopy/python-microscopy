#!/usr/bin/python

###############
# analysisTasks.py
#
# Copyright David Baddeley, 2012
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
################
# Create your views here.
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from samples.models import *
from django.http import Http404
from django import forms
from django.template import RequestContext

import Pyro.core
import Pyro.naming


class Queue:
    def __init__(self, tq, name):
        self.name = name
        self.NOpen = tq.getNumberOpenTasks(name)
        self.NInProc = tq.getNumberTasksInProgress(name)
        self.NClosed = tq.getNumberTasksCompleted(name)

class Worker:
    def __init__(self, tq, name):
        self.name = name
        self.NProc = tq.getNumTasksProcessed(name)
        self.FPS = tq.getWorkerFPS(name)

class Server:
    def __init__(self, name, ns):
        self.name = name

        try:
            tq = Pyro.core.getProxyForURI(ns.resolve('TaskQueues.%s' % name))

            queueNames = tq.getQueueNames()
            self.queues = [Queue(tq, name) for name in queueNames]

            workerNames = tq.getWorkerNames()
            self.workers = [Worker(tq, name) for name in workerNames]

            self.NProc = tq.getNumTasksProcessed()
            self.FPS = 'N/A'
        except:
            self.queues = []
            self.workers = []
            self.NProc = 0
            self.FPS = 0

            self.name = name + ' - DEAD'


def analysisTasks(request):
    try:
        from PYME.misc import pyme_zeroconf 
        ns = pyme_zeroconf.getNS()
    except:
        ns=Pyro.naming.NameServerLocator().getNS()
        
    serverNames = [n for n in ns.list('TaskQueues')]

    servers = [Server(n, ns) for n in serverNames]


    return render_to_response('samples/analysisTasks.html', {'servers':servers,})


    
