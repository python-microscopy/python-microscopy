import os.path
__author__="david"
__date__ ="$7/12/2010 3:51:15 PM$"

import glob
import os

mods = list(set([os.path.splitext(os.path.split(p)[-1])[0] for p in glob.glob(__path__[0] + '/[a-zA-Z]*.py*')]))

def InitPlugins(visFr):
    
    for mn in mods:
        m = __import__('PYME.Analysis.LMVis.Extras.' + mn, fromlist=['PYME', 'Analysis', 'LMVis', 'Extras'])
        
        m.Plug(visFr)
