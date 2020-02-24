import sqlite3
import time
import tempfile
import os
import collections

NSInfo = collections.namedtuple('NSInfo', ('name', 'address', 'port', 'creation_time', 'URI'))

class SQLiteNS(object):
    """This spoofs (but does not fully re-implement) a Pyro.naming.Nameserver using a locally held sqlite database
    
    In this case we are simply using sqlite as a key-value store which handles concurrent access across processes.
    """
    
    def __init__(self, protocol='_pyme-sql'):
        self._protocol = protocol
        self._dbname = os.path.join(tempfile.gettempdir(), '%s.sqlite' %self._protocol)
        self._conn = sqlite3.connect(self._dbname)

        tableNames = [a[0] for a in self._conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]
        if not 'dns' in tableNames:
            self._conn.execute("CREATE TABLE dns (name TEXT, address TEXT, port INTEGER, creation_time FLOAT, URI TEXT)")
    
    def register(self, name, URI):
        """ This only exists for principally for pyro compatibility - use register_service for non pyro uses
        Takes a Pyro URI object
        """
        self._conn.execute("INSERT INTO dns VALUES(?, ?, ?, ?, ?)", (name, URI.address, URI.port, time.time(), str(URI)))
        self._conn.commit()
    
    # @property
    # def advertised_services(self):
    #     return self.listener.advertised_services
    
    def get_advertised_services(self):
        names = [r[0] for r in self._conn.execute("SELECT DISTINCT name FROM dns").fetchall()]
        services = [(n, (self._conn.execute("SELECT * FROM dns WHERE name=? ORDER BY creation_time DESC ", (n,)).fetchone())) for n in names]
        return services
    
    def register_service(self, name, address, port, desc={}, URI=''):
        self._conn.execute("INSERT INTO dns VALUES(?, ?, ?, ?, ?)", (name, address, port, time.time(), URI))
        self._conn.commit()
    
    def unregister(self, name):
        self._conn.execute("DELETE FROM dns WHERE name=? ", (name,))
        self._conn.commit()
    
    def resolve(self, name):
        """ mainly for PYRO compatibility - returns a string version of the URI"""
        uri = self._conn.execute("SELECT URI FROM dns WHERE name=? ORDER BY creation_time DESC ", (name,)).fetchone()
        
        return uri[0]
    
    def list(self, filterby=''):
        return [r[0] for r in self._conn.execute("SELECT DISTINCT name FROM dns").fetchall()]
    
    def __del__(self):
        try:
            self._conn.close()
        except:
            pass


nsd = {}


def getNS(protocol='_pyme-pyro'):
    try:
        ns = nsd[protocol]
    except KeyError:
        ns = SQLiteNS(protocol)
        nsd[protocol] = ns
        #time.sleep(1) #wait for the services to come up
    
    return ns