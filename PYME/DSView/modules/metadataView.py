#!/usr/bin/python
##################
# metadata.py
#
# Copyright David Baddeley, 2011
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
from PYME.ui import MetadataTree

def Plug(dsviewer):
    mdv = MetadataTree.MetadataPanel(dsviewer, dsviewer.image.mdh)
    dsviewer.AddPage(page=mdv, select=False, caption='Metadata')

    dsviewer.mdv = mdv