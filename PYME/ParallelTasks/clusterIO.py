# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 10:07:11 2016

@author: david
"""

import PYME.misc.pyme_zeroconf as pzc
#import urllib
import socket
import requests
import time
import numpy as np

ns = pzc.getNS('_pyme-http')

def locateFile(filename, serverfilter=''):
    locs = []
    
    dirname = '/'.join(filename.split('/')[:-1])
    fn = filename.split('/')[-1]
    if (len(dirname) >=1):
        dirname = dirname + '/'
    
    for name, info in ns.advertised_services.items():
        if serverfilter in name:
            dirurl = 'http://%s:%d/%s' %(socket.inet_ntoa(info.address), info.port, dirname) 
            t = time.time()
            r = requests.get(dirurl.encode())
            dt = time.time() - t
            #print dt
            dirList = r.json()
            
            #print r.status_code
            #print dirList
            
            if fn in dirList:
                locs.append((dirurl + fn, dt))
                
    return locs

def _chooseLocation(locs):
    '''Chose the location to load the file from

    default to choosing the "closest" (currently the one with the shortest response 
    time to our directory query)
    
    '''
    
    cost = np.array([l[1] for l in locs])
    
    return locs[cost.argmin()][0]
    
def getFile(filename, serverfilter=''):
    locs = locateFile(filename)
    
    if (len(locs) ==0):
        #we did not find the file
        raise IOError("Specified file could not be found")
    else:
        url = _chooseLocation(locs)
        r = requests.get(url.encode())
        
        return r.content

_lastwritetime = {}
_lastwritespeed = {}
#_diskfreespace = {}

def _chooseServer(serverfilter=''):
    '''chose a server to save to by minimizing a cost function
    
    currently takes the server which has been waiting longest
    
    TODO: add free disk space and improve metrics/weightings
    
    '''
    serv_candidates = [(k, v) for k, v in ns.advertised_services.items() if serverfilter in k]

    t = time.time()    
    
    costs = []
    for k, v in serv_candidates:
        try:
            cost = _lastwritetime[k] - t
        except KeyError:
            cost = -100
            
#        try:
#            cost -= 0*_lastwritespeed[k]/1e3
#        except KeyError:
#            pass
        
        #try:
        #    cost -= _lastwritespeed[k]
        #except KeyError:
        #    pass
            
        costs.append(cost)# + .1*np.random.normal())
        
    return serv_candidates[np.argmin(costs)]
        
def putFile(filename, data, serverfilter=''):
    '''put a file to a server in the cluster (chosen by algorithm)
    '''
    name, info = _chooseServer(serverfilter)
    
    url = 'http://%s:%d/%s' %(socket.inet_ntoa(info.address), info.port, filename)
    
    t = time.time()
    r = requests.put(url.encode(), data=data)
    dt = time.time() - t
    #print r.status_code
    if not r.status_code == 200:
        raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))
        
    _lastwritetime[name] = t
    _lastwritespeed[name] = len(data)/dt
        
def putFiles(filenames, datas, serverfilter=''):
    '''put a bunch of files to a single server in the cluster (chosen by algorithm)
    '''
    name, info = _chooseServer(serverfilter)
    
    for filename, data in zip(filenames, datas):    
        url = 'http://%s:%d/%s' %(socket.inet_ntoa(info.address), info.port, filename)
        
        t = time.time()
        r = requests.put(url.encode(), data=data)
        dt = time.time() - t
        #print r.status_code
        if not r.status_code == 200:
            raise RuntimeError('Put failed with %d: %s' % (r.status_code, r.content))
            
        _lastwritetime[name] = t
        _lastwritespeed[name] = len(data)/dt
    
    
        
    
        
    