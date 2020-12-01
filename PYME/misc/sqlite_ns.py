import sqlite3
import time
import tempfile
import os
import collections
import socket

NSInfo = collections.namedtuple('NSInfo', ('name', 'address', 'port', 'creation_time', 'URI'))
def make_info(info):
    """Translate our results into something that looks like ``zeroconf.ServiceInfo`` - specifically convert the ip address
    into the expected format"""
    name, address, port, creation_time, URI = info
    return NSInfo(name, socket.inet_aton(address), port, creation_time, URI)

def is_port_open(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    try:
        s.connect((ip, int(port)))
        s.shutdown(socket.SHUT_RDWR)
        return True
    except:
        return False
    finally:
        s.close()


class SQLiteNS(object):
    """This spoofs (but does not fully re-implement) a Pyro.naming.Nameserver using a locally held sqlite database
    
    In this case we are simply using sqlite as a key-value store which handles concurrent access across processes.
    """
    
    def __init__(self, protocol='_pyme-sql'):
        self._protocol = protocol
        self._dbname = os.path.join(tempfile.gettempdir(), '%s.sqlite' %self._protocol)
        with sqlite3.connect(self._dbname) as conn:

            tableNames = [a[0] for a in conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]
            if not 'dns' in tableNames:
                try:
                    conn.execute("CREATE TABLE dns (name TEXT, address TEXT, port INTEGER, creation_time FLOAT, URI TEXT)")
                except sqlite3.OperationalError as e:
                    # catch race condition where table is created in another process
                    if 'table dns already exists' in str(e):
                        pass
                    else:
                        raise
                
        self.remove_inactive_services()
    
    def register(self, name, URI):
        """ This only exists for principally for pyro compatibility - use register_service for non pyro uses
        Takes a Pyro URI object
        """
        with sqlite3.connect(self._dbname) as conn:
            conn.execute("INSERT INTO dns VALUES(?, ?, ?, ?, ?)", (name, URI.address, URI.port, time.time(), str(URI)))
            conn.commit()
    
    # @property
    # def advertised_services(self):
    #     return self.listener.advertised_services
    
    def get_advertised_services(self):
        with sqlite3.connect(self._dbname) as conn:
            names = [r[0] for r in conn.execute("SELECT DISTINCT name FROM dns").fetchall()]
            services = [(n, make_info(conn.execute("SELECT * FROM dns WHERE name=? ORDER BY creation_time DESC ", (n,)).fetchone())) for n in names]
        
        return services
    
    def register_service(self, name, address, port, desc={}, URI=''):
        with sqlite3.connect(self._dbname) as conn:
            conn.execute("INSERT INTO dns VALUES(?, ?, ?, ?, ?)", (name, address, port, time.time(), URI))
            conn.commit()
    
    def unregister(self, name):
        with sqlite3.connect(self._dbname) as conn:
            conn.execute("DELETE FROM dns WHERE name=? ", (name,))
            conn.commit()
    
    def resolve(self, name):
        """ mainly for PYRO compatibility - returns a string version of the URI"""
        with sqlite3.connect(self._dbname) as conn:
            uri = conn.execute("SELECT URI FROM dns WHERE name=? ORDER BY creation_time DESC ", (name,)).fetchone()
        
        return uri[0]
    
    def list(self, filterby=''):
        with sqlite3.connect(self._dbname) as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT name FROM dns").fetchall()]
    
    def remove_inactive_services(self):
        #test to see if we can open the port, if not, remove
        for name, info in self.get_advertised_services():
            if not is_port_open(socket.inet_ntoa(info.address), info.port):
                self.unregister(name)
            


nsd = {}

import threading

sqlite_ns_lock = threading.Lock()

def getNS(protocol='_pyme-pyro'):
    #TODO - cache connections per thread?
    with sqlite_ns_lock:
        try:
            ns = nsd[protocol]
        except KeyError:
            ns = SQLiteNS(protocol)
            nsd[protocol] = ns
            #time.sleep(1) #wait for the services to come up
    
    return ns