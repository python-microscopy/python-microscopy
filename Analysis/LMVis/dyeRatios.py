'''Placeholder module for a more complicated storage of dye ratios, including
different splitter configurations.

currently just uses values for the default splitter config'''

ratios = {'A647':0.85, 'A680':0.85, 'A750': 0.11}

def getRatio(dye, mdh=None):
    if dye in ratios.keys():
        return ratios[dye]
    else:
        return None