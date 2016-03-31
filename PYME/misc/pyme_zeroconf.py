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
'''pyme_zeroconf.py

This implements a decentralized nameserver for PYRO based on using the zeroconf (aka Bonjour)
automated service discovery protocol
'''


import zeroconf as zc
import socket
import time
import Pyro.core


class ZCListener(object): 
    def __init__(self, protocol='_pyme-pyro'):
        self._protocol = protocol
        self.advertised_services = {}
    
    def remove_service(self, zeroconf, _type, name):
        #print("Service %s removed" % (name,))
        nm = name.split('.' + self._protocol)[0]
        try:
            self.advertised_services.pop(nm)
        except KeyError:
            pass
        
    def add_service(self, zeroconf, _type, name):
        #print _type, name
        nm = name.split('.' + self._protocol)[0]
        info = zeroconf.get_service_info(_type, name)
        self.advertised_services[nm] = info
        
            
class ZeroConfNS(object):
    '''This spoofs (but does not fully re-implement) a Pyro.naming.Nameserver'''
    def __init__(self, protocol = '_pyme-pyro'):
        self._services = {}
        self._protocol = protocol
        self.zeroconf = zc.Zeroconf()
        self.listener = ZCListener(self._protocol)
        
        self.browser = zc.ServiceBrowser(self.zeroconf, "%s._tcp.local." % self._protocol, 
                                         self.listener)
                                        
    def register(self, name, URI):
        desc = {'URI': str(URI)}
        
        self.register_service(name, URI.address, URI.port, desc)
        
    @property
    def advertised_services(self):
        return self.listener.advertised_services
        
    def register_service(self, name, address, port, desc={}):
        if name in self.listener.advertised_services.keys():
            raise RuntimeError('Name "%s" already exists' %name)
        
        info = zc.ServiceInfo("%s._tcp.local." % self._protocol,
                           "%s.%s._tcp.local." % (name, self._protocol),
                           socket.inet_aton(address), port, 0, 0,
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
        
nsd = {}

def getNS(protocol = '_pyme-pyro'):
    
    try:
        ns = nsd[protocol]
    except KeyError:
        ns = ZeroConfNS(protocol)
        nsd[protocol] = ns
        time.sleep(1) #wait for the services to come up
        
    return ns