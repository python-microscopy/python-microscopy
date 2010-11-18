#!/usr/bin/python

##################
# RemoteSpectrometer.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python
import Pyro.core
import Pyro.naming

import com.oceanoptics.omnidriver.api.wrapper.Wrapper

class RemoteSpectrometer(com.oceanoptics.omnidriver.api.wrapper.Wrapper, Pyro.core.ObjBase):
    def __init__(self):
        com.oceanoptics.omnidriver.api.wrapper.Wrapper.__init__(self)
        Pyro.core.ObjBase.__init__(self)

        self.openAllSpectrometers()



if __name__ == '__main__':

	Pyro.config.PYRO_MOBILE_CODE = 1
	Pyro.core.initServer()
	ns=Pyro.naming.NameServerLocator().getNS()
	daemon=Pyro.core.Daemon()
	daemon.useNameServer(ns)

	dd = RemoteSpectrometer()
	uri=daemon.connect(dd,"USB2000p")

	try:
		daemon.requestLoop()
	finally:
		daemon.shutdown(True)