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

from imp import reload

from PYME import config

localmodules = [os.path.splitext(os.path.split(p)[-1])[0] for p in glob.glob(__path__[0] + '/[a-zA-Z]*.py')]

modLocations = {}
for m in localmodules:
    modLocations[m] = ['PYME', 'DSView', 'modules']

for m in config.get_plugins('dsviewer'):
    ms = m.split('.')
    modLocations[ms[-1]] = ms[:-1]


def allmodules():
    am = modLocations.keys()
    am.sort()
    
    return am


basemodules = ['shell', 'metadataView', 'eventView', 'deconvolution', 'tiling', 'recipes']
liteModules = ['filtering', 'cropping','composite', 'profilePlotting', 'splitter', 'synchronise']

modeModules = {
'lite': ['arrayView'] + liteModules,
'LM' : ['arrayView'] + liteModules + basemodules +  ['LMAnalysis'],
'blob' : ['arrayView'] + liteModules + basemodules + ['blobFinding', 'psfExtraction'],
'default' : ['arrayView'] + liteModules + basemodules,
'psf'   :   ['arrayView'] + liteModules + ['deconvolution', 'psfTools'],
'visGUI' : ['visgui'] + liteModules + ['coloc', 'vis3D'],
'graph' : ['graphViewPanel', 'profileFitting'],
'fgraph' : ['fastGraphPanel'],
'pupil' : ['arrayView', 'pupilTools'] + liteModules,
'tracking' : ['arrayView'] + liteModules + basemodules + ['particleTracking'],
'bare' : [],
}

def loadModule(modName, dsviewer):
    ml = modLocations[modName]
    mod = __import__('.'.join(ml) + '.' + modName, fromlist=ml)
    
    #if modName in dsviewer.installedModules:
    try: 
        mod.Unplug(dsviewer)
    except:
        pass
    
    reload(mod)
        
    mod.Plug(dsviewer)
    

    dsviewer.installedModules.append(modName)


def loadMode(mode, dsviewer):
    """install the relevant modules for a particular mode"""

    if mode in modeModules.keys():
        mods = modeModules[mode]
    else:
        mods = modeModules['default']

    for m in mods:
        loadModule(m, dsviewer)
