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
import glob
import os
from PYME import config
import importlib

# discover PYME-internal fitfactories and fully qualify them
__all__ = ['PYME.localization.FitFactories.' + os.path.splitext(os.path.split(p)[-1])[0] for p in glob.glob(__path__[0] + '/[a-zA-Z]*.py')]
# find plug-in fitfactories
__all__.extend(list(config.get_plugins('fit_factories')))

#fitFactoryList = glob.glob(PYME.localization.FitFactories.__path__[0] + '/[a-zA-Z]*.py')
#fitFactoryList = [os.path.split(p)[-1][:-3] for p in fitFactoryList]
#fitFactoryList.sort()

resFitFactories = []
descriptions = {}
longDescriptions = {}
useFor = {}
for ff in __all__:
    try:
        fm = importlib.import_module(ff, '.'.join(ff.split('.')[:-1]))
        ff = ff.split('.')[-1]
        if 'FitResultsDType' in dir(fm):
            resFitFactories.append(ff)
            if 'DESCRIPTION' in dir(fm):
                descriptions[ff] = fm.DESCRIPTION
                longDescriptions[ff] = fm.LONG_DESCRIPTION
            else:
                descriptions[ff] = ''
                longDescriptions[ff] =''
            if 'USE_FOR' in dir(fm):
                useFor[ff] = fm.USE_FOR
            else:
                useFor[ff] = ''
    except:
        pass
    
resFitFactories.sort()
