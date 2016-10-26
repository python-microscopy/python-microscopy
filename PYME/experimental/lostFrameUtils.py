
##################
# eventFilterUtils.py
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

def idTransientFrames(dataSource, events, fps):
    """
    Adds a 'isTransient' column to the input datasource, so that localizations from frames which were acquired during
    z-translation can be selectively filtered
    Args:
        dataSource: dictionary or recarray
        events: dictionary-like object
        fps: frames acquired per step

    Returns:
        nothing, but adds column to input datasource

    """
    # need this next bit to work if events is a dictionary or a recarray:
    names = np.array(events['EventName'])
    descr = np.array(events['EventDescr'])
    focusChanges = descr[names == 'ProtocolFocus']

    # for ProtocolFocus events, description is 'frame#, position'
    t = np.copy(dataSource['t'])
    ti = np.ones_like(t, dtype=int)

    for fi in focusChanges:
        fi = float(fi.split(',')[0])  # only interested in frame# at the moment
        ti[np.logical_and(t >= fi, t < fi+fps)] -= 1

    dataSource.addColumn('isTransient', ti)
    return