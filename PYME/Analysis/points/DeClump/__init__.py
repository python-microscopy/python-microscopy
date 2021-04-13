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
from . import deClump

# wrap c functions so we can get nice docstrings.
#################################


def findClumps(t, x, y, delta_x, n_frames=10, link_within_frame=True):
    """
    Find repeated localisations of a single fluorophore, connecting localisations which are within `2*delta_x`
    (i.e. a 95% confidence interval) of a localisation in the previous frame. Handles intermittency
    (rapid blinking etc ...) through the `n_frames` parameter.

    By making `delta_x` half the maximum movement of a particle from one frame to the next, the code can also be used
    for sparse particle tracking.
    
    Localisations **MUST BE SORTED** in time-order prior to calling findClumps.
    
    Parameters
    ----------
    t : ndarray (int)
        frame numbers for localisations
    x : ndarray (float)
        x positions of localisations
    y : ndarray (float)
        y positions of localisations
    delta_x: ndarray (float)
        localisation error (std. dev.) for each localisation
    n_frames: int
        number of frames an emitter can be off due to transient blinking and still be linked. 0 = must be on in previous
        frame, 1 = can have a gap of 1 frame, etc ... The special value of -1 only links within the current frame.
    link_within_frame: bool
        whether to link within a frame. If True (default to match old behaviour) close emitters within a frame will be
        linked.

    Returns
    -------

    assigned : ndarray (int)
        array of labels for each linked chunk found
        
    
    Notes
    -----
    
    When events are so dense that multiple emitters in a single frame are within `2*delta_x`, the linking will also be
    a bit dumb, and might link traces incorrectly. In practice this is only likely to occur in a single-particle
    tracking scenario where `delta_x` has been inflated to allow for potential emitter movement. In these cases, feature
    based tracking, which can use other object properties such as event intensity and/or shape to improve linkages
    is preferred.

    We also assume that `delta_x` is a good proxy for `delta_y` - the actual linkage function used is
    $(x_{i+1}-x_i)^2 + (y_{i+1} - y_i)^2 < 4 \times delta_x^2$. This is a fair assumption for the majority of
    localisation data.
    """
    
    return deClump.findClumps(t, x, y, delta_x, n_frames, link_within_frame)