
##################
# flagZMotionArtifacts.py
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

import logging

logger = logging.getLogger(__name__)




class motionFlagger:
    """
    Class with methods to flag localizations for filtering with respect to information stored in the events table
    """
    def __init__(self, visFr):
        self.pipeline = visFr.pipeline


        logging.debug('Adding menu items for event filters')

        visFr.AddMenuItem('Corrections', 'Identify transient frames', self.OnIDTransient,
                          helpText='Toss frames acquired during pifoc translation')


    def OnIDTransient(self, event=None):
        """
        Adds an 'isTransient' column to the input datasource so that one can filter localizations that are from frames
        acquired during z-translation
        """
        from PYME.experimental import zMotionArtifactUtils
        mask = zMotionArtifactUtils.flagMotionArtifacts(self.pipeline.selectedDataSource, self.pipeline.events,
                                           self.pipeline.mdh['StackSettings.FramesPerStep'])
                                           
        self.pipeline.selectedDataSource.addColumn('piezoUnstable', mask)
        #eventFilterUtils.idTransientFrames(self.pipeline.selectedDataSource, self.pipeline.events,
        #                                   self.pipeline.mdh['StackSettings.FramesPerStep'])

def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.eventfilters = motionFlagger(visFr)

