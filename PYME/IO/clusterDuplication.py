# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Mon Mar 28 16:56:03 2016

This module handles data duplication on the cluster and ensures that there are 
always at least N copies of a file present. Current default is 2, but 3 might 
make more sense in the long term 

@author: david
"""

from . import clusterIO
import Pyro.core

import PYME.misc.pyme_zeroconf as pzc

import time
import numpy as np

import threading

import os

from PYME.misc.computerName import GetComputerName
compName = GetComputerName()
procName = compName + ' - PID:%d' % os.getpid()

ns = pzc.getNS('_pyme-pyro')

TARGETCOPIES = 2


def duplicateFiles(targetNCopies=TARGETCOPIES, serverfilter=clusterIO.local_serverfilter):
    """walks the cluster and mirrors any files with a count less than the target
    
    for high duplication targets (e.g. NCopies = 3), each pass will only
    increment the copy number by 1
    """

    ncopied = 0  
    
    #print targetNCopies, serverfilter
    
    for top, dirs, files in clusterIO.walk('', serverfilter=serverfilter):
        for f in files:
            #print (top+f),len(clusterIO.locateFile(top + f)) 
            if len(clusterIO.locate_file(top + f)) < targetNCopies:
                print('Duplicating %s' % (top + f))
                clusterIO.mirror_file(top + f)
                ncopied += 1
                
    print('%d files duplicated' % ncopied)
    

class DupManager(Pyro.core.ObjBase):
    """ This class handles electing a master duplicator such that only one
    client is managing duplication at any given time.
    
    We use the "Bully" algorithm for leader election
    """
    
    def __init__(self, targetNCopies=TARGETCOPIES, serverfilter=clusterIO.local_serverfilter):
        Pyro.core.ObjBase.__init__(self)
        self._targetNCopies = targetNCopies
        self._serverfilter = serverfilter
        
        self._leader = None
        self.name = 'PYMEDupClient - ' + procName
        pass
    
    def ping(self, nodename):
        ret = False
        try:
            nd = Pyro.core.getProxyForURI(ns.resolve(nodename))
            nd._setTimeout(1)
        
            ret = nd.is_leader()          
        except:
            pass
        
        return ret
        
    
    def is_leader(self):
        return self._leader == self.name
        
    def assert_leader(self, leader):
        self._leader = leader
        
    def elect(self):
        #find all DupClient nodes
        nodes = ns.list('DupClient')
        
        #we haven't found a higher priority node before we look
        higher_node = False        
        
        for n in nodes: 
            if (n > self.name):
                #loop through higher priority nodes and try to elect them
                #node priority is determined by a > comparison on the name string
                ret = False
                try:
                    nd = Pyro.core.getProxyForURI(ns.resolve(n))
                    nd._setTimeout(1)
                
                    ret = nd.elect()
                except:
                    pass
                
                higher_node = higher_node or ret
                
                if ret:
                    return True
                    
        if not higher_node:
            #no higher priority nodes responded
            #we are the new leader
            
            self._leader = self.name
            
            print('%s is the new leader' % self._leader)
            
            #loop through all nodes and crown ourselves
            for n in nodes:
                if not (n == self.name): #don't try and call ourselves
                    try:
                        nd = Pyro.core.getProxyForURI(ns.resolve(n))
                        nd._setTimeout(1)
                    
                        ret = nd.assert_leader(self._leader)
                    except:
                        pass
                    
        return True
        
    def attempt_dup(self):
        if (self._leader is None) or not self.ping(self._leader):
            #no leader or leader unreachable, trigger a new election
            self.elect()
            
            #we will return at this point in case a previous duplication from 
            #another master is running, actual processing will occur in next call
            #to this function
            return False
            
        if self.is_leader():
            #only actually initiate a duplication if we are the leader
            duplicateFiles(self._targetNCopies, self._serverfilter)
            return True
        else:
            return False
            


keepAlive = True
def keep_alive():
    return keep_alive
    
def _requestLoop(daemon):
    try:
        daemon.requestLoop(keep_alive)
    finally:
        daemon.shutdown(True)
    
DELAY_S = 10.

def startDuplicator():
    daemon=Pyro.core.Daemon()
    daemon.useNameServer(ns)
    
    dup = DupManager()
    
    daemon.connect(dup,dup.name)
    
    #start a new thread to handle remote access to our duplicator    
    t = threading.Thread(target=_requestLoop, args=(daemon,))
    t.start()

    try:    
        while keepAlive:
            #delay for a semi-random ammount of time
            time.sleep(DELAY_S + np.random.rand()*DELAY_S)
            
            #attempt to duplicate
            dup.attempt_dup()
    finally:
        stopDuplicator()
        
    
    
        
def stopDuplicator():
    global keepAlive
    keepAlive = False

if __name__ == '__main__':
    startDuplicator()
        
    
            
                
        