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
import os.path

import glob
import os

from PYME import config
from PYME.DSView.modules import _load_mod

import logging

logger = logging.getLogger(__name__)

mods = list(set([os.path.splitext(os.path.split(p)[-1])[0] for p in glob.glob(__path__[0] + '/[a-zA-Z]*.py') + glob.glob(__path__[0] + '/[a-zA-Z]*.pyc')]))
mods.sort()


def InitPlugins(visFr):
    for mn in mods:
        #print mods
        logger.debug('Initializing %s plugin' % mn)
        m = __import__('PYME.LMVis.Extras.' + mn, fromlist=['PYME', 'LMVis', 'Extras'])
        #m.Plug(visFr)
        _load_mod(m, mn, visFr)

    for mn in config.get_plugins('visgui'):
        try:
            m = __import__(mn, fromlist=mn.split('.')[:-1])
            #m.Plug(visFr)
            _load_mod(m, mn, visFr)
        except Exception as e:
            #import traceback
            #traceback.print_exc()
            logger.exception('Error loading plugin: %s' % mn)
