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

import logging
logger = logging.getLogger(__name__)

# discover PYME-internal fitfactories
_pyme_ff = [os.path.splitext(os.path.split(p)[-1])[0] for p in glob.glob(__path__[0] + '/[a-zA-Z]*.py')]
_pyme_ff.sort() #sort builtin and plugins before combining so that builtins always appear at the top.
_pyme_ff_pkg = 'PYME.localization.FitFactories'

# find plug-in fitfactories
_plugin_ff = list(config.get_plugins('fit_factories'))
_plugin_ff.sort()

### Note:
#
# - Built in fit modules will appear using just the module name (stripped of 'PYME.localization.FitFactories')
# - Fit factories from plugins will use a fully resolved module path (i.e. myplugin.myfitmodule)
#
# This is to allow easy identification of who to contact for support and to reduce the potential for naming conflicts.

resFitFactories = []
descriptions = {}
longDescriptions = {}
useFor = {}
package = {}

def _register_fit_factory(key, module_name, pkg=_pyme_ff_pkg):
    try:
        #get fit module info
        fm = importlib.import_module(pkg + '.' + module_name)
        #package[key] = pkg
        if 'FitResultsDType' in dir(fm):
            resFitFactories.append(key)
            if 'DESCRIPTION' in dir(fm):
                descriptions[key] = fm.DESCRIPTION
                longDescriptions[key] = fm.LONG_DESCRIPTION
            else:
                descriptions[key] = ''
                longDescriptions[key] =''
            if 'USE_FOR' in dir(fm):
                useFor[key] = fm.USE_FOR
            else:
                useFor[key] = ''
    except:
        logger.exception('Error registering fit factory: %s, %s, %s' % (key, module_name, pkg))
        pass

for ff in _pyme_ff:
    _register_fit_factory(ff, ff, _pyme_ff_pkg)

for ff in _plugin_ff:
    parts = ff.split('.')
    mod_name = parts[-1]
    pkg = '.'.join(parts[:-1])
    _register_fit_factory(ff, mod_name, pkg)
    

def import_fit_factory(fit_factory):
    """Helper function to import a fit factory from just the module name,
    resolves the location and does the import, returning the module

    Parameters
    ----------
    fit_factory : str
        fit factory module name, e.g. LatGaussFitFR

    Returns
    -------
    module
        loaded fit factory module
    """
    
    # create fully resolved path for builtin modules
    if fit_factory in _pyme_ff:
        fit_factory = _pyme_ff_pkg + '.' + fit_factory

    return importlib.import_module(fit_factory)
