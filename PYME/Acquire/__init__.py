#!/usr/bin/python

###############
# __init__.py
#
# Copyright David Baddeley, 2012
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
################
"""
This package contains all the functionality of the `PYMEAcquire` acquisition program.

The core components are:

- :py:mod:`PYME.Acquire.PYMEAcquire` : the GUI entry point
- :py:mod:`PYME.Acquire.acquiremainframe` : the GUI code for the main window

- :py:mod:`PYME.Acquire.microscope` : a handler / collection point for all the hardware and microscope state
- :py:mod:`PYME.Acquire.frameWrangler` : controls the flow of data from the camera(s)

Most of the additional modules serve a supporting role. Of special note are the spoolers (:py:mod:`PYME.IO.HDFSpooler`,
:py:mod:`PYME.IO.QueueSpooler`, and :py:mod:`PYME.IO.HTTPSpooler`) which are the backends for spooling data acquisition.

Drivers for different pieces of experimental hardware are found in :py:mod:`PYME.Acquire.Hardware`



`PYMEAcquire` can be launched by running the `PYMEAcquire` script, specifying an initialization file to be used. For more
info see :py:mod:`PYME.Acquire.PYMEAcquire`.

"""