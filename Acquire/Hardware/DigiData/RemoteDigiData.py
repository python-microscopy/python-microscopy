#!/usr/bin/python
from DigiData import *
import Pyro.core
import Pyro.naming

class RemoteDigiData(DigiData, Pyro.core.ObjBase):
    def __init__(self):
        DigiData.__init__(self)
        Pyro.core.ObjBase.__init__(self)

if __name__ == '__main__':

	Pyro.config.PYRO_MOBILE_CODE = 1
	Pyro.core.initServer()
	ns=Pyro.naming.NameServerLocator().getNS()
	daemon=Pyro.core.Daemon()
	daemon.useNameServer(ns)

	dd = RemoteDigiData()
	uri=daemon.connect(dd,"DigiData")

	try:
		daemon.requestLoop()
	finally:
		daemon.shutdown(True)
