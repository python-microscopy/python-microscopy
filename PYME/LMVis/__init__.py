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
This package contains code related to VisGUI, the visualization program for localization point data sets. The main
components would be:
  - :py:mod:`PYME.LMVis.VisGUI` : the main GUI code used for point visualization
  - :py:mod:`PYME.LMVis.pipeline` : the underlying pipeline object which handles file opening and manipulation (this is
                                  the core non-gui component of point processing)

Also of interest are:
  - :py:mod:`PYME.LMVis.inpFilt` : The filters and mappings used to build up the pipeline
  - :py:mod:`PYME.LMVis.renderers` : The code which interfaces the individual rendering methods used to convert point
                                   data to some kind of density map.

Plugins are located in :py:mod:`PYME.LMVis.Extras`.

The VisGUI can be launched using the `VisGUI` script.

"""
