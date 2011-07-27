__author__="david"
__date__ ="$3/02/2011 9:42:00 PM$"

import glob
import os

allmodules = [os.path.splitext(os.path.split(p)[-1])[0] for p in glob.glob(__path__[0] + '/[a-zA-Z]*.py')]


basemodules = ['shell', 'metadataView', 'eventView', 'deconvolution', 'tiling']
liteModules = ['composite', 'profilePlotting']

modeModules = {
'lite': ['arrayView'] + liteModules,
'LM' : ['arrayView'] + liteModules + basemodules +  ['LMAnalysis'],
'blob' : ['arrayView'] + liteModules + basemodules + ['blobFinding', 'psfExtraction'],
'default' : ['arrayView'] + liteModules + basemodules + ['psfExtraction'],
'visGUI' : liteModules + ['coloc', 'vis3D'],
'graph' : ['graphViewPanel']
}

def loadModule(modName, dsviewer):
    mod = __import__('PYME.DSView.modules.' + modName, fromlist=['PYME', 'DSView', 'modules'])
    mod.Plug(dsviewer)

    dsviewer.installedModules.append(modName)


def loadMode(mode, dsviewer):
    '''install the relevant modules for a particular mode'''

    if mode in modeModules.keys():
        mods = modeModules[mode]
    else:
        mods = modeModules['default']

    for m in mods:
        loadModule(m, dsviewer)