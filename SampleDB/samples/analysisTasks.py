# Create your views here.
from django.shortcuts import render_to_response, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from SampleDB.samples.models import *
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
    ns=Pyro.naming.NameServerLocator().getNS()
    serverNames = [n[0] for n in ns.list('TaskQueues')]

    servers = [Server(n, ns) for n in serverNames]


    return render_to_response('samples/analysisTasks.html', {'servers':servers,})


    
