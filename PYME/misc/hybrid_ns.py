from . import sqlite_ns
from . import pyme_zeroconf

class HybridNS(object):
    """This spoofs (but does not fully re-implement) a Pyro.naming.Nameserver using a both zeroconf and a locally held
    sqlite database. It's principle use case is as a catch-all for client programs which don't care how the servers advertise.
    """
    
    def __init__(self, protocol='_pyme-sql'):
        self._protocol = protocol
        self._zc_ns = pyme_zeroconf.getNS(protocol)
        
    @property
    def _sqlite_ns(self):
        #get this on the fly to work around sqlite threading issues (is cached on a per-thread basis in the sqlite_ns module)
        return sqlite_ns.getNS(self._protocol)
    
    def register(self, name, URI):
        """ This only exists for principally for pyro compatibility - use register_service for non pyro uses
        Takes a Pyro URI object
        """
        #register with both
        self._zc_ns.register(name, URI)
        self._sqlite_ns.register(name, URI)
    
    # @property
    # def advertised_services(self):
    #     return self.listener.advertised_services
    
    def get_advertised_services(self):
        services = dict(self._sqlite_ns.get_advertised_services())
        services.update(dict(self._zc_ns.get_advertised_services()))
        return list(services.items())
    
    def register_service(self, name, address, port, desc={}, URI=''):
        """

        Parameters
        ----------
        name : str
            Max 63 chars long due to zeroconf requirements. PYME.IO.FileUtils.nameUtils.get_service_name can be used to get a suitable (truncated if necessary)
            name.

        """
        self._zc_ns.register_service(name, address, desc)
        self._sqlite_ns.register_service(name, address, desc, URI)
    
    def unregister(self, name):
        """

        Parameters
        ----------
        name : str
            must be the same service name used to register

        """
        self._zc_ns.unregister(name)
        self._sqlite_ns.unregister(name)
    
    def resolve(self, name):
        """ mainly for PYRO compatibility - returns a string version of the URI"""
        try:
            uri = self._sqlite_ns.resolve(name)
        except:
            uri = self._zc_ns.resolve(name)
            
        return uri
    
    def list(self, filterby=''):
        return list(set(self._sqlite_ns.list() + list(self._zc_ns.list())))
    

nsd = {}


def getNS(protocol='_pyme-pyro'):
    try:
        ns = nsd[protocol]
    except KeyError:
        ns = HybridNS(protocol)
        nsd[protocol] = ns
        #time.sleep(1) #wait for the services to come up
    
    return ns
