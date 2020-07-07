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
import weakref

from imp import reload

from PYME import config
import logging
logger = logging.getLogger(__name__)

localmodules = [os.path.splitext(os.path.split(p)[-1])[0] for p in glob.glob(__path__[0] + '/[a-zA-Z]*.py')]

modLocations = {}
for m in localmodules:
    modLocations[m] = ['PYME', 'DSView', 'modules']

for m in config.get_plugins('dsviewer'):
    ms = m.split('.')
    modLocations[ms[-1]] = ms[:-1]


def allmodules():
    am = list(modLocations.keys())
    am.sort()
    
    return am


basemodules = ['shell', 'metadataView', 'eventView', 'deconvolution', 'tiling', 'recipes', 'vis3D']
liteModules = ['filtering', 'cropping','composite', 'profilePlotting', 'splitter', 'synchronise']

modeModules = {
'lite': ['arrayView'] + liteModules,
'LM' : ['arrayView'] + liteModules + basemodules +  ['LMAnalysis'],
'blob' : ['arrayView'] + liteModules + basemodules + ['blobFinding', 'psfExtraction'],
'default' : ['arrayView'] + liteModules + basemodules,
'psf'   :   ['arrayView'] + liteModules + ['deconvolution', 'psfTools', 'metadataView'],
'visGUI' : ['visgui'] + liteModules + ['coloc', 'vis3D'],
'graph' : ['graphViewPanel', 'profileFitting'],
'fgraph' : ['fastGraphPanel'],
'pupil' : ['arrayView', 'pupilTools'] + liteModules,
'tracking' : ['arrayView'] + liteModules + basemodules + ['particleTracking'],
'bare' : [],
}

def loadModule(modName, dsviewer):
    """
    Loads a module my calling that modules `Plug()` function.
    
    The `outputs = Plug(dsviewer)` function takes an instance of the data viewer and optionally
    returns a module class (or classes) which should be programatically accessible as an attribute
    of the viewer. This return can either be an object (in which case it will be accessible under
    the plugin name) or a dictionary if multiple objects are being returned or if things should be
    available under a name other than the module name. The dictionary return is largely to support
    an easy transition from the old plugin naming and should be avoided in new code, using a wrapper
    class if more than one object needs to be accessed. In the future we might require the return to
    be a subclass of a `Plugin` object or similar.
    
    Plugins should **NOT** inject themselves directly into the dsviewer namespace as has been done
    in the case in the past as this is likely to result in circular references.
    
    Plugins should also avoid holding a reference to dsviewer.image, dsviewer.do, etc, although it is
    currently permissible to hold a reference to dsviewer itself TODO - Move these into a plugin base
    class and use weakrefs.
    
    
    Parameters
    ----------
    modName
    dsviewer

    Returns
    -------

    """
    ml = modLocations[modName]
    mod = __import__('.'.join(ml) + '.' + modName, fromlist=ml)
    
    #if modName in dsviewer.installedModules:
    try: 
        mod.Unplug(dsviewer)
    except:
        pass
    
    reload(mod)
    
    # record dsviewer keyw so we can warn if we inject into dsviewer class
    # part of module injection deprecation
    dsv_keys = list(dsviewer.__dict__.keys())
    ret = mod.Plug(dsviewer)
    
    if list(dsviewer.__dict__.keys()) != dsv_keys:
        logger.warning('Plugin [%s] injects into dsviewer namespace, could result in circular references' % modName)

    # new style module tracking
    # TODO - make this accessible via dsviewer.plugins.x rather than dsviewer.x
    if ret is not None:
        if isinstance(ret, dict):
            # either track named outputs [legacy]
            for k, v in ret:
                setattr(dsviewer, k, weakref.proxy(v))
        else:
            # or track module data under module name.
            setattr(dsviewer, modName, weakref.proxy(ret))
            #dsviewer._module_injections[modName] = ret


    dsviewer.installedModules.append(modName)


def loadMode(mode, dsviewer):
    """install the relevant modules for a particular mode"""

    if mode in modeModules.keys():
        mods = modeModules[mode]
    else:
        mods = modeModules['default']

    for m in mods:
        loadModule(m, dsviewer)
