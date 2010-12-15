'''Placeholder module for a more complicated storage of dye ratios, including
different splitter configurations.

currently just uses values for the default splitter config'''

ratios = {'A647':0.85, 'A680':0.87, 'A750': 0.11, 'CF770': 0.11}

def getRatio(dye, mdh=None):
    if dye in ratios.keys():
        if 'Splitter.TransmittedPathPosition' in mdh.getEntryNames() and mdh.getEntry('Splitter.TransmittedPathPosition') == 'Top':
            return 1 - ratios[dye]
        else:
            return ratios[dye]
    else:
        return None