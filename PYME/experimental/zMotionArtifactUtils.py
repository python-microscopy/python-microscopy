
##################
# zMotionArtifactUtils.py
#
# Copyright David Baddeley, Andrew Barentine
# david.baddeley@yale.edu
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

import numpy as np

def flagMotionArtifacts(dataSource, events, fps):
    """
    flags frames where we think the piezo might be moving so we can filter them out later
    
    NOTE: Only works for a non-standard usage of ProtocolFocus
    
    Args:
        dataSource: dictionary or recarray
        events: dictionary-like object
        fps: frames acquired per step

    Returns:
        array with mask for frames without potential motion artifacts

    """
    # need this next bit to work if events is a dictionary or a recarray:
    names = np.array(events['EventName'])
    descr = np.array(events['EventDescr'])
    focusChanges = descr[names == 'ProtocolFocus']

    # for ProtocolFocus events, description is 'frame#, position'
    t = dataSource['t']
    ti = np.ones_like(t, dtype=int)

    # TODO: note that this works for a broken implementation of ProtocolFocus, where the ProtocolFocus event is written after piezo settling rather than on initiation of movement
    for fi in focusChanges:
        fi = float(fi.split(',')[0])  # only interested in frame# at the moment
        ti[np.logical_and(t >= fi, t < fi+fps)] -= 1

    #dataSource.addColumn('isTransient', ti)
    return ti