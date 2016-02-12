#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2016
# 
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
'''pyro_zeroconf.py

This implements a decentralized nameserver for PYRO based on using the zeroconf (aka Bonjour)
automated service discovery protocol
'''


import zeroconf as zc
import socket
import time
import Pyro.core


class ZCListener(object):
    advertised_services = {}
    
    def remove_service(self, zeroconf, _type, name):
        #print("Service %s removed" % (name,))
        nm = name.split('._pyme-pyro')[0]
        try:
            self.advertised_services.pop(nm)
        except KeyError:
            pass
        
    def add_service(self, zeroconf, _type, name):
        #print _type, name
        nm = name.split('._pyme-pyro')[0]
        info = zeroconf.get_service_info(_type, name)
        self.advertised_services[nm] = info
        
            
class ZeroConfNS(object):
    '''This spoofs (but does not fully re-implement) a Pyro.naming.Nameserver'''
    def __init__(self):
        self._services = {}
        self.zeroconf = zc.Zeroconf()
        self.listener = ZCListener()
        
        self.browser = zc.ServiceBrowser(self.zeroconf, "_pyme-pyro._tcp.local.", 
                                         self.listener)
                                        
    def register(self, name, URI):
        if name in self.listener.advertised_services.keys():
            raise RuntimeError('Name "%s" already exists' %name)
        
        desc = {'URI': str(URI)}
        
        #print URI.address, type(URI.address), URI.port, type(URI.port)
        
        info = zc.ServiceInfo("_pyme-pyro._tcp.local.",
                           "%s._pyme-pyro._tcp.local." % name,
                           socket.inet_aton(URI.address), URI.port, 0, 0,
                           desc)
                           
        self._services[name] = info
        self.zeroconf.register_service(info)
        
    def unregister(self, name):
        try:
            info = self._services[name]
            self.zeroconf.unregister_service(info)
        except KeyError:
            raise KeyError('Name "%s" not registered on this computer' %name)
            
    def resolve(self, name):
        #try:
        info = self.listener.advertised_services[name]
        return info.properties['URI']
        
    def list(self, filterby = ''):
        return [k for k in self.listener.advertised_services.keys() if filterby in k]
        
            
    def __del__(self):
        self.zeroconf.unregister_all_services()
        self.zeroconf.close()
        
ns = None

def getNS():
    global ns
    if ns == None:
        ns = ZeroConfNS()
        
    return ns