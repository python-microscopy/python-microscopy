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
"""pyme_zeroconf.py

This implements a decentralized nameserver for PYRO based on using the zeroconf (aka Bonjour)
automated service discovery protocol
"""


import zeroconf
import socket
import time
import Pyro.core

import threading

class PatchedServiceInfo(zeroconf.ServiceInfo):
    """
    A patched version of the zero-conf ServiceInfo object which ensures that we get the port number as well as the address and name
    """
    
    def request(self, zc, timeout):
        """Returns true if the service could be discovered on the
        network, and updates this object with details discovered.
        """
        now = zeroconf.current_time_millis()
        delay = zeroconf._LISTENER_TIME
        next = now + delay
        last = now + timeout
        result = False
        try:
            zc.add_listener(self, zeroconf.DNSQuestion(self.name, zeroconf._TYPE_ANY, zeroconf._CLASS_IN))
            while None in (self.server, self.address,self.text): #, self.port):
                if last <= now:
                    return False
                if next <= now:
                    out = zeroconf.DNSOutgoing(zeroconf._FLAGS_QR_QUERY)
                    out.add_question(zeroconf.DNSQuestion(self.name, zeroconf._TYPE_SRV,
                                                          zeroconf._CLASS_IN))
                    out.add_answer_at_time(zc.cache.get_by_details(self.name,
                                                                   zeroconf._TYPE_SRV, zeroconf._CLASS_IN), now)
                    out.add_question(zeroconf.DNSQuestion(self.name, zeroconf._TYPE_TXT,
                                                 zeroconf._CLASS_IN))
                    out.add_answer_at_time(zc.cache.get_by_details(self.name,
                                                                   zeroconf._TYPE_TXT, zeroconf._CLASS_IN), now)
                    if self.server is not None:
                        out.add_question(zeroconf.DNSQuestion(self.server,
                                                              zeroconf._TYPE_A, zeroconf._CLASS_IN))
                        out.add_answer_at_time(zc.cache.get_by_details(self.server,
                                                                       zeroconf._TYPE_A, zeroconf._CLASS_IN), now)
                    zc.send(out)
                    next = now + delay
                    delay = delay * 2

                zc.wait(min(next, last) - now)
                now = zeroconf.current_time_millis()
            result = True
        finally:
            zc.remove_listener(self)

        return result

class ZCListener(object): 
    def __init__(self, protocol='_pyme-pyro'):
        self._protocol = protocol
        self.advertised_services = {}
        
        self._lock = threading.Lock()
    
    def remove_service(self, zc, _type, name):
        #print("Service %s removed" % (name,))
        nm = name.split('.' + self._protocol)[0]
        try:
            with self._lock:
                self.advertised_services.pop(nm)
        except KeyError:
            pass
        
    def add_service(self, zc, _type, name):
        #print _type, name
        nm = name.split('.' + self._protocol)[0]
        
        with self._lock:
            #info = zc.get_service_info(_type, name)
            info = PatchedServiceInfo(_type, name)
            if info.request(zc, 3000):
                self.advertised_services[nm] = info

    def list(self, filterby):
        with self._lock:
            return [k for k in self.advertised_services.keys() if filterby in k]
        
    def get_info(self, name):
        with self._lock:
            return self.advertised_services[name]
        
    def get_advertised_services(self):
        with self._lock:
            return list(self.advertised_services.items())
        
            
class ZeroConfNS(object):
    """This spoofs (but does not fully re-implement) a Pyro.naming.Nameserver"""
    def __init__(self, protocol = '_pyme-pyro'):
        self._services = {}
        self._protocol = protocol
        self.zc = zeroconf.Zeroconf()
        self.listener = ZCListener(self._protocol)
        
        self.browser = zeroconf.ServiceBrowser(self.zc, "%s._tcp.local." % self._protocol,
                                         self.listener)
                                        
    def register(self, name, URI):
        desc = {'URI': str(URI)}
        
        self.register_service(name, URI.address, URI.port, desc)
        
    # @property
    # def advertised_services(self):
    #     return self.listener.advertised_services
    
    def get_advertised_services(self):
        return self.listener.get_advertised_services()
        
    def register_service(self, name, address, port, desc={}):
        if name in self.listener.advertised_services.keys():
            raise RuntimeError('Name "%s" already exists' %name)
        
        info = PatchedServiceInfo("%s._tcp.local." % self._protocol,
                           "%s.%s._tcp.local." % (name, self._protocol),
                           socket.inet_aton(address), port, 0, 0,
                           desc)
                           
        self._services[name] = info
        self.zc.register_service(info)
        
    def unregister(self, name):
        try:
            info = self._services[name]
            self.zc.unregister_service(info)
        except KeyError:
            raise KeyError('Name "%s" not registered on this computer' %name)
            
    def resolve(self, name):
        #try:
        info = self.listener.get_info(name)
        return info.properties['URI']
        
    def list(self, filterby = ''):
        return self.listener.list(filterby)
        
            
    def __del__(self):
        try:
            self.zc.unregister_all_services()
            self.zc.close()
        except:
            pass
        
nsd = {}

def getNS(protocol = '_pyme-pyro'):
    
    try:
        ns = nsd[protocol]
    except KeyError:
        ns = ZeroConfNS(protocol)
        nsd[protocol] = ns
        time.sleep(1) #wait for the services to come up
        
    return ns