#!/usr/bin/jython

##################
# RemoteSpectrometer.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import Pyro.core
import Pyro.naming

import com.oceanoptics.omnidriver.api.wrapper.Wrapper

class RemoteSpectrometer(com.oceanoptics.omnidriver.api.wrapper.Wrapper, Pyro.core.ObjBase):
    def __init__(self):
        com.oceanoptics.omnidriver.api.wrapper.Wrapper.__init__(self)
        Pyro.core.ObjBase.__init__(self)

        self.openAllSpectrometers()



if __name__ == '__main__':

    Pyro.config.PYRO_MOBILE_CODE = 0
    Pyro.core.initServer()
    ns=Pyro.naming.NameServerLocator().getNS()
    daemon=Pyro.core.Daemon()
    daemon.useNameServer(ns)

    #get rid of any previous queue
    try:
        ns.unregister("USB2000p")
    except Pyro.errors.NamingError:
        pass

    dd = RemoteSpectrometer()
    uri=daemon.connect(dd,"USB2000p")

    try:
        daemon.requestLoop()
    finally:
        daemon.shutdown(True)