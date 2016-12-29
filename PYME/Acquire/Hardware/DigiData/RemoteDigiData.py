#!/usr/bin/python

##################
# RemoteDigiData.py
#
# Copyright David Baddeley, 2009
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

#!/usr/bin/python
from .DigiData import *
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

